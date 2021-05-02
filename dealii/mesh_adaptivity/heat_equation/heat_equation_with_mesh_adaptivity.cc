#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <fstream>
#include <iostream>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/base/std_cxx17/cmath.h>
#include <deal.II/base/geometric_utilities.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/distributed/solution_transfer.h>

namespace Poisson
{
  using namespace dealii;
  
  template <int dim>
  class FunctionRHS : public Function <dim>
  {
  public:
    FunctionRHS(double time=0) : Function<dim>() {this->time=time;}
    virtual double value (const Point<dim> &p, const unsigned int component=0) const;
    virtual void set_time (const double time){this->time=time;};
    double time;
  };
  
  template <int dim>
  double FunctionRHS<dim>::value (const Point<dim> &p, const unsigned int) const
  {
    double x = p[0];
    double y = p[1];
    return (2*M_PI*(4*M_PI*std::cos(2*M_PI*time) - std::sin(2*M_PI*time))
	    *std::sin(2*M_PI*x)*std::sin(2*M_PI*y));
  }

  template <int dim>
  class InitCond : public Function <dim>
  {
  public:
    InitCond() : Function<dim>() {}
    virtual double value (const Point<dim> &p, const unsigned int component=0) const;
  };
  
  template <int dim>
  double InitCond<dim>::value (const Point<dim> &p, const unsigned int) const
  {
    double x = p[0];
    double y = p[1];
    return std::sin(2*M_PI*x)*std::sin(2*M_PI*y);
  }

  template <int dim>
  class ExactSolution : public Function <dim>
  {
  public:
    ExactSolution() : Function<dim>() {}
    virtual double value (const Point<dim> &p, const unsigned int component=0) const;
  };
  
  template <int dim>
  double ExactSolution<dim>::value (const Point<dim> &p, const unsigned int) const
  {
    double x = p[0];
    double y = p[1];
    return std::sin(2*M_PI*x)*std::sin(2*M_PI*y);
  }

  template <int dim>  
  class HeatEquation
  
  {
  public:
    HeatEquation(const unsigned int degree);
    void run();
    
  private:
    void get_initial_condition();
    void setup_system();
    void refine_mesh(const unsigned int min_grid_level,
		     const unsigned int max_grid_level);
    double get_dt();
    void evolve_to_time(const double final_time);
    void evolve_one_time_step();	
    void assemble_matrix();
    void assemble_matrices();
    void assemble_rhs();
    void solve();
    void output_results(const unsigned int cycle) const;
    void get_errors();	
	
    MPI_Comm mpi_communicator;

    // domain and grid
    parallel::distributed::Triangulation<dim> triangulation;

    // numerics
    bool use_iterative_solver;
    unsigned int degree;
    FE_Q<dim>   fe;
    DoFHandler<dim> dof_handler;
    MappingQ<dim> mapping;
        
    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;
    std::map< types::global_dof_index, Point<dim> > support_points;
	
    AffineConstraints<double> constraints;

    PETScWrappers::MPI::SparseMatrix system_matrix, mass_matrix, stiffness_matrix;
    std::shared_ptr<PETScWrappers::PreconditionBlockJacobi> preconditioner;
    PETScWrappers::MPI::Vector       un,unp1;
    PETScWrappers::MPI::Vector       system_rhs;

    // utilities
    ConditionalOStream pcout;

    // time related quantities
    double time;
    double final_time;
    double dt;
    double cfl;
    double min_cell_diameter;
    unsigned int verbosity;
  };

  template <int dim>
  HeatEquation<dim>::HeatEquation(const unsigned int degree)
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
							       Triangulation<dim>::smoothing_on_refinement |
							       Triangulation<dim>::smoothing_on_coarsening))
    , degree(degree)
    , fe(degree)
    , dof_handler(triangulation)
    , mapping(MappingQ<dim>(degree,true)) // high-order mapping in the interior and boundary elements
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  {}

  template<int dim>
  void HeatEquation<dim>::get_initial_condition()
  {
    PETScWrappers::MPI::Vector completely_distributed_init_condition(locally_owned_dofs, mpi_communicator);
    VectorTools::interpolate(mapping,
			     dof_handler, 
			     InitCond<dim>(), 
			     completely_distributed_init_condition);
    unp1 = completely_distributed_init_condition; // to output the init condition
    un = completely_distributed_init_condition; // to output the init condition
  }

  template<int dim>
  void HeatEquation<dim>::evolve_to_time(const double final_time)
  {
    bool final_step = false;
    while (time<final_time)
      {
	if (verbosity==1)
	  pcout << "           Time: " << time << std::endl;
	evolve_one_time_step();

	// update old solution
	un = unp1;
		
	// check if this is the last step
	if (final_step==true)
	  {
	    // adjust the time for the final step and exit
	    time += dt; 
	    break;
	  }
	else
	  {
	    // update time
	    time += dt; 

	    // check if the next step is the last one
	    if (time+dt >= final_time)
	      {
		final_step=true;
		// adjust the final time step size
		dt = final_time - time;
		assemble_matrix();
	      }
	  }
      }

    // refinie the mesh
    refine_mesh(4,6);
    un = unp1;
	
    // adjust dt 
    dt = get_dt();
    assemble_matrix();
  }

  template<int dim>
  void HeatEquation<dim>::evolve_one_time_step()
  {
    assemble_rhs();
    solve();
  }

  template <int dim>
  void HeatEquation<dim>::refine_mesh(const unsigned int min_grid_level,
				      const unsigned int max_grid_level)
  {
    // Get locally relevant solution for u
    PETScWrappers::MPI::Vector locally_relevant_unp1;
    locally_relevant_unp1.reinit(locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
    locally_relevant_unp1 = unp1;

    // Get error estimate
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(dof_handler,
				       QGauss<dim-1>(fe.degree + 1),
				       std::map<types::boundary_id,
				       const Function<dim> *>(),
				       locally_relevant_unp1,
				       estimated_error_per_cell);

    // set refinement factors
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
									     estimated_error_per_cell,
									     0.5,
									     0.5);
    if (triangulation.n_levels() > max_grid_level)
      for (const auto &cell : triangulation.active_cell_iterators_on_level(max_grid_level))
	cell->clear_refine_flag();
    for (const auto &cell : triangulation.active_cell_iterators_on_level(min_grid_level))
      cell->clear_coarsen_flag();

    // solution transfer
    parallel::distributed::SolutionTransfer<dim,PETScWrappers::MPI::Vector> solution_trans_unp1(dof_handler);
    PETScWrappers::MPI::Vector previous_unp1;
    previous_unp1.reinit(locally_owned_dofs, mpi_communicator);
    previous_unp1 = unp1;

    // prepare_coarsening_and_refinement for the triangulation 
    triangulation.prepare_coarsening_and_refinement();
		
    // prepare_coarsening_and_refinement for the SolutionTransfer for the velocity
    solution_trans_unp1.prepare_for_coarsening_and_refinement(previous_unp1);
	
    // execute_coarsening_and_refinement();
    triangulation.execute_coarsening_and_refinement();
	
    //setup_system();
    setup_system();

    // interpolate the solution for the pressure
    solution_trans_unp1.interpolate(unp1);

    // distribute constraints
    constraints.distribute(unp1);
	
    // other adjustments
    dt = get_dt();
	
    // compute again the stiffness matrix for the Poisson solve
    assemble_matrix();
  }
  
  template <int dim>
  void HeatEquation<dim>::setup_system()
  {
    // LOCALLY OWNED AND LOCALLY RELEVANT DOFs //
    // distributes DoF in parallel
    dof_handler.distribute_dofs(fe); 
    locally_owned_dofs = dof_handler.locally_owned_dofs(); 
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs); // these include ghost cells

    // RHS AND SOLUTION //
    un.reinit(locally_owned_dofs, mpi_communicator);
    unp1.reinit(locally_owned_dofs, mpi_communicator); 
    system_rhs.reinit(locally_owned_dofs, mpi_communicator); // the RHS only needs info locally owned

    // CONSTRAINTS //
    // The next step is to compute constraints like Dirichlet BCs and hanging nodes
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
					     0,
					     Functions::ZeroFunction<dim>(),
					     constraints);
    constraints.close();
	
    // initializing the matrix with sparsity pattern.
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false); // keep_constrained_dofs=false
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               dof_handler.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs);
    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
    mass_matrix.reinit(locally_owned_dofs,
		       locally_owned_dofs,
		       dsp,
		       mpi_communicator);
    stiffness_matrix.reinit(locally_owned_dofs,
			    locally_owned_dofs,
			    dsp,
			    mpi_communicator);
	
    DoFTools::map_dofs_to_support_points(mapping,
					 dof_handler,
					 support_points);
  }

  template <int dim>
  void HeatEquation<dim>::assemble_matrix()
  {
    system_matrix = 0.;	
    // create a quadrature rule
    // Recall that 2*Nq-1>=degree => Nq>=(degree+1)/2
    const QGauss<dim> quadrature_formula(fe.degree + 1);

    // computation of quantities related to FEs
    FEValues<dim> fe_values(mapping,
			    fe,
			    quadrature_formula,
			    update_values |
			    update_gradients |
			    update_JxW_values);	
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();
	  
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<double> shape_value(dofs_per_cell);
    std::vector<Tensor<1,dim> > shape_grad(dofs_per_cell);
	
    // FE loop
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_matrix = 0.;
		  
	  // get shape functions, their derivatives, etc at quad points
	  fe_values.reinit(cell); 
	  
	  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
	      // compute detJxW, phi, grad(phi), etc at quad points
	      const double detJxdV = fe_values.JxW(q_point);
	      for (unsigned int i = 0; i < dofs_per_cell; ++i)
		{
		  shape_value[i] = fe_values.shape_value(i,q_point);
		  shape_grad[i]  = fe_values.shape_grad(i,q_point);
		}
	      
	      // loop on i-DoFs
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
		  // loop on j-DoFs
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
		    cell_matrix(i,j) += (shape_value[i] * shape_value[j] 
					 +
					 dt * (shape_grad[i] * shape_grad[j])
					 ) * detJxdV;
                }
            }
	  // assemble from local to global operators
          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 local_dof_indices,
                                                 system_matrix);
        }
    // Distribute between processors
    system_matrix.compress(VectorOperation::add);
    preconditioner=NULL;
    preconditioner.reset(new PETScWrappers::PreconditionBlockJacobi(system_matrix));
  }

  template <int dim>
  void HeatEquation<dim>::assemble_matrices()
  {
    mass_matrix = 0.;
    stiffness_matrix = 0.;	
    // create a quadrature rule
    // Recall that 2*Nq-1>=degree => Nq>=(degree+1)/2
    const QGauss<dim> quadrature_formula(fe.degree + 1);

    // computation of quantities related to FEs
    FEValues<dim> fe_values(mapping,
			    fe,
			    quadrature_formula,
			    update_values |
			    update_gradients |
			    update_JxW_values);	
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();
	  
    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<double> shape_value(dofs_per_cell);
    std::vector<Tensor<1,dim> > shape_grad(dofs_per_cell);
	
    // FE loop
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_mass_matrix = 0.;
	  cell_stiffness_matrix = 0.;
		  
	  // get shape functions, their derivatives, etc at quad points
	  fe_values.reinit(cell); 
	  
	  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
	      // compute detJxW, phi, grad(phi), etc at quad points
	      const double detJxdV = fe_values.JxW(q_point);
	      for (unsigned int i = 0; i < dofs_per_cell; ++i)
		{
		  shape_value[i] = fe_values.shape_value(i,q_point);
		  shape_grad[i]  = fe_values.shape_grad(i,q_point);
		}
	      
	      // loop on i-DoFs
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
		  // loop on j-DoFs
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
		    {
		      cell_mass_matrix(i,j) += shape_value[i] * shape_value[j] * detJxdV;
		      cell_stiffness_matrix(i,j) += shape_grad[i] * shape_grad[j] * detJxdV;
		    }
                }
            }
	  // assemble from local to global operators
          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_mass_matrix,
                                                 local_dof_indices,
                                                 mass_matrix);
	  constraints.distribute_local_to_global(cell_stiffness_matrix,
                                                 local_dof_indices,
                                                 stiffness_matrix);
        }
    // Distribute between processors
    mass_matrix.compress(VectorOperation::add);
    stiffness_matrix.compress(VectorOperation::add);
  }  

  template <int dim>
  void HeatEquation<dim>::assemble_rhs()
  {
    system_rhs = 0.;

    PETScWrappers::MPI::Vector locally_relevant_un;
    locally_relevant_un.reinit(locally_owned_dofs,
			       locally_relevant_dofs,
			       mpi_communicator);
    locally_relevant_un = un;
    // create a quadrature rule
    // Recall that 2*Nq-1>=degree => Nq>=(degree+1)/2
    const QGauss<dim> quadrature_formula(fe.degree + 1);
	
    // computation of quantities related to FEs
    FEValues<dim> fe_values(mapping,
			    fe,
			    quadrature_formula,
			    update_values |
			    update_gradients |
			    update_quadrature_points |
			    update_JxW_values);	
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();
	
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<double> uh_at_xq(n_q_points);
	
    FunctionRHS<dim> function;
    function.set_time(time+dt);
	
    // FE loop
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
	{
	  cell_rhs    = 0.;
		  
	  // get shape functions, their derivatives, etc at quad points
	  fe_values.reinit(cell); 
	  fe_values.get_function_values(locally_relevant_un,uh_at_xq);
		  
	  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
	    {
	      // compute detJxW, phi, grad(phi), etc at quad points
	      const double detJxdV = fe_values.JxW(q_point);
			  
	      const double rhs_value
		= uh_at_xq[q_point] + dt * function.value(fe_values.quadrature_point(q_point));
			  
	      // loop on i-DoFs
	      for (unsigned int i = 0; i < dofs_per_cell; ++i)
		cell_rhs(i) += rhs_value * fe_values.shape_value(i,q_point) * detJxdV;
	    }
		  
	  // ***** loop on faces ***** //
	  // non needed for a simple projection
		  
	  // assemble from local to global operators
	  cell->get_dof_indices(local_dof_indices);
	  constraints.distribute_local_to_global(cell_rhs,
						 local_dof_indices,
						 system_rhs);
	}
    // Distribute between processors
    system_rhs.compress(VectorOperation::add);
  }
	
  template <int dim>
  void HeatEquation<dim>::solve()
  {
    PETScWrappers::MPI::Vector completely_distributed_solution(locally_owned_dofs,
							       mpi_communicator);
    completely_distributed_solution = 0.;
    SolverControl solver_control(dof_handler.n_dofs(), 1e-12);
    use_iterative_solver=true;
    if (use_iterative_solver)
      {
	PETScWrappers::SolverCG solver(solver_control, mpi_communicator);
	solver.solve(system_matrix,
		     completely_distributed_solution,
		     system_rhs,
		     *preconditioner);
      }
    else
      {
	PETScWrappers::SparseDirectMUMPS solver(solver_control,mpi_communicator);
	solver.solve(system_matrix,
		     completely_distributed_solution,
		     system_rhs);		
      }
    if (verbosity==1)
      pcout << "   Solved in " << solver_control.last_step() << " iterations." << std::endl;

    // apply constraints
    constraints.distribute(completely_distributed_solution);
    unp1 = completely_distributed_solution;
  }

  template <int dim>
  void HeatEquation<dim>::output_results(const unsigned int cycle) const
  {
    PETScWrappers::MPI::Vector locally_relevant_output_vector;
    locally_relevant_output_vector.reinit(locally_owned_dofs,
					  locally_relevant_dofs,
					  mpi_communicator);
		
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> solution_names(1);
    solution_names[0] = "u";
    locally_relevant_output_vector = unp1;
    data_out.add_data_vector(locally_relevant_output_vector, solution_names);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(mapping,degree,DataOut<dim>::no_curved_cells);
    data_out.write_vtu_with_pvtu_record("./", "solution", cycle, mpi_communicator, 2, 8);
  }
  
  template <int dim>
  void HeatEquation<dim>::get_errors()
  {
    PETScWrappers::MPI::Vector locally_relevant_unp1;
    locally_relevant_unp1.reinit(locally_owned_dofs,
				 locally_relevant_dofs,
				 mpi_communicator);
    locally_relevant_unp1 = unp1;
    Vector<double> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(mapping,
				      dof_handler,
				      locally_relevant_unp1,
				      ExactSolution<dim>(),
				      difference_per_cell,
				      QGauss<dim>(fe.degree + 1),
				      VectorTools::L2_norm);
    const double L2_error =
      VectorTools::compute_global_error(triangulation,
					difference_per_cell,
					VectorTools::L2_norm);
    pcout << "   L2 error: "
	  << L2_error
	  << std::endl;
  }

  template <int dim>
  double HeatEquation<dim>::get_dt()
  {
    min_cell_diameter = GridTools::minimal_cell_diameter(triangulation);
    dt = cfl*min_cell_diameter;
    return dt;
  }
  
  template <int dim>
  void HeatEquation<dim>::run()
  {
    pcout << "Running with "
          << "PETSc"
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    // ***** OUTPUT CONTROLLER ***** //
    final_time=1.0;
    double output_time = 0.1;
	
    // Number of output files
    unsigned int nOut = int(final_time/output_time)+1;
    nOut += (nOut-1)*output_time < final_time ? 1 : 0;
    // create the tnList
    std::vector<double> tnList(nOut);
    for (unsigned int i=0; i<nOut; ++i)
      tnList[i] = i*output_time <= final_time ? i*output_time : final_time;

    unsigned int refinement = 4; // initial refinement
    // ***** Initial time ***** //
    time = 0.0;
	
    // ***** DOMAIN ***** /
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(refinement);
	
    // ***** SETUP ***** //
    setup_system();
    pcout << "   Number of active cells:       "
	  << triangulation.n_global_active_cells() << std::endl
	  << "   Number of degrees of freedom: " << dof_handler.n_dofs()
	  << std::endl;
	
    // ***** INITIAL CONDITIONS ***** //
    get_initial_condition();
    // output initial condition //
    {
      output_results(0);
    }
	
    // more numerical parameters
    min_cell_diameter = GridTools::minimal_cell_diameter(triangulation);
    cfl=0.5;
    dt = get_dt();
		
    // ***** ASSEMBLE MASS MATRIX ***** //
    assemble_matrix();
	
    // ***** TIME LOOP ***** //
    // loop over tnList
    for (unsigned int time_interval=1; time_interval<nOut; ++time_interval)
      {
	pcout << "***** Evolve solution from time "
	      << tnList[time_interval-1]
	      << " to time "
	      << tnList[time_interval]
	      << std::endl;
			
	evolve_to_time(tnList[time_interval]);
		    
	// output solution //
	{
	  output_results(time_interval);
	}
      } // end of time_interval
    get_errors();
  } //run
}  //
	
int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Poisson;
      
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      
      unsigned int degree = 1;
      HeatEquation<2> poisson_problem_2d(degree);
      poisson_problem_2d.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
		<< std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      std::cerr << "Exception on processing: " << std::endl
		<< exc.what() << std::endl
		<< "Aborting!" << std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
		<< std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      std::cerr << "Unknown exception!" << std::endl
		<< "Aborting!" << std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      return 1;
    }
  
  return 0;
}

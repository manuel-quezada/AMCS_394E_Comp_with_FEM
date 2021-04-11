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
#include <deal.II/base/geometric_utilities.h>

namespace Poisson
{
  using namespace dealii;
  
  template <int dim>
  class FunctionRHS : public Function <dim>
  {
  public:
    FunctionRHS() : Function<dim>() {}
    virtual double value (const Point<dim> &p, const unsigned int component=0) const;
  };
  
  template <int dim>
  double FunctionRHS<dim>::value (const Point<dim> &p, const unsigned int) const
  {
	std::array<double,dim> polar;
	polar = GeometricUtilities::Coordinates::to_spherical(p);
	double r = polar[0];
	double theta = polar[1];
	double pi = M_PI;
	double omega = 4;
	
	return -1.0/r/r*(pi*r*std::cos(pi*r)
					 - (pi*pi*r*r+omega*omega)*std::sin(pi*r))*std::cos(omega*theta);
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
	std::array<double,dim> polar;
	polar = GeometricUtilities::Coordinates::to_spherical(p);
	double r = polar[0];
	double theta = polar[1];
	double pi = M_PI;
	double omega = 4;

	return std::sin(pi*r)*std::cos(omega*theta);
  }

  template <int dim>
  class NeumannBC : public Function <dim>
  {
  public:
    NeumannBC() : Function<dim>() {}
    virtual double value (const Point<dim> &p, const unsigned int component=0) const;
  };
  
  template <int dim>
  double NeumannBC<dim>::value (const Point<dim> &p, const unsigned int) const
  {
	std::array<double,dim> polar;
	polar = GeometricUtilities::Coordinates::to_spherical(p);
	double theta = polar[1];
	double pi = M_PI;
	double omega = 4;

	return -pi*std::cos(omega*theta);
  }

  template <int dim>  
  class PoissonProblem
  {
  public:
    PoissonProblem(const unsigned int degree);
    void run();
    
  private:
    void setup_system();
    void assemble_system();
    void solve();
    void output_results(const unsigned int cycle) const;
    void get_errors(const unsigned int cycle);	
	
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

    AffineConstraints<double> constraints;

    PETScWrappers::MPI::SparseMatrix system_matrix;
    PETScWrappers::MPI::Vector       locally_relevant_solution;
    PETScWrappers::MPI::Vector       system_rhs;

    // utilities
    ConditionalOStream pcout;
    ConvergenceTable convergence_table;	
	
    // physical parameters

    // boundary conditions
  };

  template <int dim>
  PoissonProblem<dim>::PoissonProblem(const unsigned int degree)
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

  template <int dim>
  void PoissonProblem<dim>::setup_system()
  {
    // LOCALLY OWNED AND LOCALLY RELEVANT DOFs //
    // distributes DoF in parallel
    dof_handler.distribute_dofs(fe); 
    locally_owned_dofs = dof_handler.locally_owned_dofs(); 
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs); // these include ghost cells

    // RHS AND SOLUTION //
    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator); // the RHS only needs info locally owned

    // CONSTRAINTS //
    // The next step is to compute constraints like Dirichlet BCs and hanging nodes
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
	VectorTools::interpolate_boundary_values(mapping,
											 dof_handler,
											 0,
											 ExactSolution<dim>(),
											 constraints);
	//VectorTools::interpolate_boundary_values(dof_handler,
	//										 1,
	//										 ExactSolution<dim>(),
	//										 constraints);
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
  }

  template <int dim>
  void PoissonProblem<dim>::assemble_system()
  {
    // create a quadrature rule
    // Recall that 2*Nq-1>=degree => Nq>=(degree+1)/2
    const QGauss<dim> quadrature_formula(fe.degree + 1);
	const QGauss<dim-1> face_quadrature_formula(fe.degree+1);
	
    // computation of quantities related to FEs
    FEValues<dim> fe_values(mapping,
							fe,
							quadrature_formula,
							update_values |
							update_gradients |
							update_quadrature_points |
							update_JxW_values);
	FEFaceValues<dim> fe_face_values(mapping,
									 fe,
									 face_quadrature_formula,
									 update_values | update_quadrature_points |
									 update_normal_vectors |
									 update_JxW_values);
	
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();
	const unsigned int n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<double> shape_value(dofs_per_cell);
	
    FunctionRHS<dim> function;
	NeumannBC<dim> neumann_bcs;
	  
    // FE loop
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_matrix = 0.;
          cell_rhs    = 0.;
	  
		  // get shape functions, their derivatives, etc at quad points
		  fe_values.reinit(cell); 
	  
		  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
			  // compute detJxW, phi, grad(phi), etc at quad points
			  const double detJxdV = fe_values.JxW(q_point);
			  for (unsigned int i = 0; i < dofs_per_cell; ++i)
				{
				  shape_value[i] = fe_values.shape_value(i,q_point);
				}
	      
			  const double rhs_value = function.value(fe_values.quadrature_point(q_point));
			  // loop on i-DoFs
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
				  cell_rhs(i) += rhs_value * shape_value[i] * detJxdV;
				  // loop on j-DoFs
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
					cell_matrix(i,j) += (fe_values.shape_grad(i,q_point)
										 *fe_values.shape_grad(j,q_point)) * detJxdV;
                }
            }

		  // ***** loop on faces ***** //
		  for (const auto &face : cell->face_iterators())
			if (face->at_boundary() && (face->boundary_id() == 1))
			  {
				fe_face_values.reinit(cell, face);
				for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
				  {
					const double rhs_value = neumann_bcs.value(fe_face_values.quadrature_point(q_point));
					// loop on i-DoFs
					for (unsigned int i = 0; i < dofs_per_cell; ++i)
					  {
						cell_rhs(i) += rhs_value *
						  fe_face_values.shape_value(i, q_point) * // phi_i(x_q)
						  fe_face_values.JxW(q_point);            // dx
					  }
				  }
			  }
	
		  // assemble from local to global operators
          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
        }
    // Distribute between processors
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  template <int dim>
  void PoissonProblem<dim>::solve()
  {
    PETScWrappers::MPI::Vector completely_distributed_solution(locally_owned_dofs,
															   mpi_communicator);

    SolverControl solver_control(dof_handler.n_dofs(), 1e-12);
    if (use_iterative_solver)
      {
		PETScWrappers::SolverCG solver(solver_control, mpi_communicator);
		PETScWrappers::PreconditionBoomerAMG preconditioner;
		PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
		data.symmetric_operator = true;
		preconditioner.initialize(system_matrix, data);
		solver.solve(system_matrix,
					 completely_distributed_solution,
					 system_rhs,
					 preconditioner);
      }
    else
      {
		PETScWrappers::SparseDirectMUMPS solver(solver_control,mpi_communicator);
		solver.solve(system_matrix,
					 completely_distributed_solution,
					 system_rhs);
      }
    pcout << "   Solved in " << solver_control.last_step() << " iterations." << std::endl;

    // apply constraints
    constraints.distribute(completely_distributed_solution);
    locally_relevant_solution = completely_distributed_solution;
  }

  template <int dim>
  void PoissonProblem<dim>::output_results(const unsigned int cycle) const
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> solution_names(1);
    solution_names[0] = "u";
    data_out.add_data_vector(locally_relevant_solution, solution_names);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(mapping,degree,DataOut<dim>::no_curved_cells);
    data_out.write_vtu_with_pvtu_record("./", "solution", cycle, mpi_communicator, 2, 8);
  }
  
  template <int dim>
  void PoissonProblem<dim>::get_errors(const unsigned int cycle)
  {
    Vector<double> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(mapping,
									  dof_handler,
									  locally_relevant_solution,
									  ExactSolution<dim>(),
									  difference_per_cell,
									  QGauss<dim>(fe.degree + 1),
									  VectorTools::L2_norm);
    const double L2_error =
      VectorTools::compute_global_error(triangulation,
										difference_per_cell,
										VectorTools::L2_norm);
    pcout << "   L2 error for cycle "
		  << cycle
		  << ": "
		  << L2_error
		  << std::endl;

    // save error into convergence_table
    const unsigned int n_active_cells=triangulation.n_global_active_cells();
    const unsigned int n_dofs = dof_handler.n_dofs();	
    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("n_dofs", n_dofs);
    convergence_table.add_value("E2", L2_error);
  }
  
  template <int dim>
  void PoissonProblem<dim>::run()
  {
    pcout << "Running with "
          << "PETSc"
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    // ******************************* //
    // ***** physical parameters ***** //
    // ******************************* //

    // ******************************** //
    // ***** numerical parameters ***** //
    // ******************************** //
    unsigned int refinement = 3; // initial refinement
    unsigned int num_cycles = 4;
    bool get_error=true;
    use_iterative_solver=true;
    
    for (unsigned int cycle = 0; cycle < num_cycles; ++cycle)
      {
		pcout << "Cycle " << cycle << ':' << std::endl;
		// ***** DOMAIN AND MESH ***** //
		if (cycle == 0)
		  {
			const Point<2> center (0,0);
			const double inner_radius = 0.25, outer_radius = 1.0;
			GridGenerator::hyper_shell(triangulation,
									   center, inner_radius, outer_radius,
									   0,true);
			triangulation.refine_global(refinement);
		  }
		else
		  triangulation.refine_global(1);
	
		// ***** SETUP ***** //
		setup_system();
		pcout << "   Number of active cells:       "
			  << triangulation.n_global_active_cells() << std::endl
			  << "   Number of degrees of freedom: " << dof_handler.n_dofs()
			  << std::endl;
	
		// ***** ASSEMBLE AND SOLVE SYSTEM ***** //
		assemble_system();
		solve();
	
		// ***** GET ERRORS ***** //
		if (get_error)
		  {
			get_errors(cycle);
			if (cycle==0)
			  {
				convergence_table.set_precision("E2", 2);
				convergence_table.set_scientific("E2",true);
				convergence_table.set_tex_format("cells","r");
				convergence_table.set_tex_format("n_dofs","r");
			  }
		  }
	
		// ***** OUTPUT THE RESULTS ***** //
		if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
		  output_results(cycle);
		pcout << std::endl;
      }

    // ***** GET CONVERGENCE RATES AND PRINT TABLE ***** //
    convergence_table.evaluate_convergence_rates("E2", "cells", ConvergenceTable::reduction_rate_log2, dim);
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
		std::cout << std::endl;
		convergence_table.write_text(std::cout);
      }
  }
} // namespace Poisson

int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Poisson;
      
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      
      unsigned int degree = 1;
      PoissonProblem<2> poisson_problem_2d(degree);
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

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

namespace NS
{
  using namespace dealii;
  
  template <int dim>
  class FunctionRHS_u : public Function <dim>
  {
  public:
    FunctionRHS_u(double time=0) : Function<dim>() {this->time=time;}
    virtual double value (const Point<dim> &p, const unsigned int component=0) const;
	virtual void set_time (const double time){this->time=time;};
	double time;
  };
  
  template <int dim>
  double FunctionRHS_u<dim>::value (const Point<dim> &p, const unsigned int) const
  {
    double x = p[0];
    double y = p[1];
	double mu = 1.0;
	
	return (std::cos(time+y)*std::sin(x) // time derivative
			+ 2.0*mu*std::sin(x)*std::sin(time+y) // viscosity
			+ std::cos(x)*std::sin(x) // nonlinearity
			); 
  }

  template <int dim>
  class FunctionRHS_v : public Function <dim>
  {
  public:
    FunctionRHS_v(double time=0) : Function<dim>() {this->time=time;}
    virtual double value (const Point<dim> &p, const unsigned int component=0) const;
	virtual void set_time (const double time){this->time=time;};
	double time;
  };
  
  template <int dim>
  double FunctionRHS_v<dim>::value (const Point<dim> &p, const unsigned int) const
  {
    double x = p[0];
    double y = p[1];
	double mu = 1.0;
	return (-std::sin(time+y)*std::cos(x) // time derivative
			+ 2.0*mu*std::cos(x)*std::cos(time+y) // viscosity
			+ -0.5*std::sin(2*(time+y))
			);
  }

  template <int dim>
  class ExactSolution_u : public Function <dim>
  {
  public:
    ExactSolution_u(double time) : Function<dim>() {this->time=time;}
    virtual double value (const Point<dim> &p, const unsigned int component=0) const;
	double time;
  };
  
  template <int dim>
  double ExactSolution_u<dim>::value (const Point<dim> &p, const unsigned int) const
  {	
    double x = p[0];
    double y = p[1];
	return std::sin(x)*std::sin(y+time);
  }

  template <int dim>
  class ExactSolution_v : public Function <dim>
  {
  public:
    ExactSolution_v(double time) : Function<dim>(){this->time=time;}
    virtual double value (const Point<dim> &p, const unsigned int component=0) const;
	double time;
  };
  
  template <int dim>
  double ExactSolution_v<dim>::value (const Point<dim> &p, const unsigned int) const
  {
    double x = p[0];
    double y = p[1];
	return std::cos(x)*std::cos(y+time);
  }

  template <int dim>
  class ExactSolution_P : public Function <dim>
  {
  public:
    ExactSolution_P(double time) : Function<dim>(){this->time=time;}
    virtual double value (const Point<dim> &p, const unsigned int component=0) const;
	double time;
  };
  
  template <int dim>
  double ExactSolution_P<dim>::value (const Point<dim> &p, const unsigned int) const
  {
    double x = p[0];
    double y = p[1];
	return std::cos(x)*std::sin(y+time);
  }

  template <int dim>  
  class NavierStokes
  
  {
  public:
    NavierStokes(const unsigned int degree);
    void run();
    
  private:
	void get_initial_condition();
	void set_boundary_values(double time);
	void get_boundary_ids();
    void setup_system();
	void evolve_to_time(const double final_time);
    void evolve_one_time_step();	
    void assemble_matrix();
	void assemble_rhs();
    void solve();
	void assemble_matrix_phi();
	void assemble_rhs_phi();
    void solve_phi();
    void output_results(const unsigned int cycle) const;
    void get_errors(const unsigned int cycle);	
	
    MPI_Comm mpi_communicator;

    // domain and grid
    parallel::distributed::Triangulation<dim> triangulation;

    // numerics
    bool use_iterative_solver;
    unsigned int degree;

    MappingQ<dim> mapping;

	// FE spaces for velocity
	FE_Q<dim>   fe;
    DoFHandler<dim> dof_handler;
    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

	// FE spaces for pressure
	FE_Q<dim>   fe_P;
    DoFHandler<dim> dof_handler_P;
    IndexSet locally_owned_dofs_P;
    IndexSet locally_relevant_dofs_P;

	// constraints
    AffineConstraints<double> constraints_U, constraints_phi;

    PETScWrappers::MPI::SparseMatrix system_matrix_u, system_matrix_v, system_matrix_phi;
	std::shared_ptr<PETScWrappers::PreconditionBlockJacobi> preconditioner_u, preconditioner_v, preconditioner_phi;
	PETScWrappers::MPI::Vector       phinm1,phin,phinp1;
	PETScWrappers::MPI::Vector       pn,pnp1;
	PETScWrappers::MPI::Vector       unm1,un,unp1;
	PETScWrappers::MPI::Vector       vnm1,vn,vnp1;
    PETScWrappers::MPI::Vector       system_rhs_u, system_rhs_v, system_rhs_phi;

	// for Dirichlet boundary values
	std::vector<unsigned int> boundary_values_id_u;
	std::vector<unsigned int> boundary_values_id_v;
	
    // utilities
    ConditionalOStream pcout;
    ConvergenceTable convergence_table;	

    // physical parameters
	double coeff_rho;
	double coeff_mu;
	
    // boundary conditions

	// time related quantities
    double time;
    double final_time;
    double dt;
    double cfl;
    double min_cell_diameter;
    unsigned int verbosity;
  };

  template <int dim>
  NavierStokes<dim>::NavierStokes(const unsigned int degree)
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
															   Triangulation<dim>::smoothing_on_refinement |
															   Triangulation<dim>::smoothing_on_coarsening))
    , degree(degree)
    , fe(degree)
    , dof_handler(triangulation)
	, fe_P(degree)
	, dof_handler_P(triangulation)
    , mapping(MappingQ<dim>(degree,true)) // high-order mapping in the interior and boundary elements
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  {}

  template<int dim>
  void NavierStokes<dim>::get_initial_condition()
  {
    PETScWrappers::MPI::Vector init_condition(locally_owned_dofs, mpi_communicator);
    VectorTools::interpolate(mapping,
							 dof_handler, 
							 ExactSolution_u<dim>(0.0), 
							 init_condition);
	un = init_condition;
    unm1 = init_condition; 
	unp1 = init_condition; // to output the init condition
	VectorTools::interpolate(mapping,
							 dof_handler,
							 ExactSolution_v<dim>(0.0),
							 init_condition);
	vn = init_condition;
	vnm1 = init_condition; 
    vnp1 = init_condition; // to output the init condition

	// for pressure
	PETScWrappers::MPI::Vector init_condition_P(locally_owned_dofs_P, mpi_communicator);
	VectorTools::interpolate(mapping,
							 dof_handler_P,
							 ExactSolution_P<dim>(0.0),
							 init_condition_P);
	pn = init_condition_P;
	pnp1 = init_condition_P;
	phinm1 = 0.;
	phin = 0.;
	phinp1 = 0.;
  }

  template<int dim>
  void NavierStokes<dim>::evolve_to_time(const double final_time)
  {
    bool final_step = false;
    while (time<final_time)
      {
		if (verbosity==1)
		  pcout << "           Time: " << time << std::endl;
		evolve_one_time_step();

		// update old solution
		// for u
		unm1 = un;
		un = unp1;
		
		// for v		
		vnm1 = vn;
		vn = vnp1;

		// for phi
		phinm1 = phin;
		phin = phinp1;

		// for p
		pn = pnp1;
		
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
			if (time+dt >= final_time - 1E-12)
			  {
				final_step=true;
				// no need to adjust the final time step size since dt is chosen accordingly
				//dt = final_time - time;
				//assemble_matrix();
			  }
		  }
      }
    // adjust dt. No need to adjust back dt since I don't change it anymore
    //dt = cfl*min_cell_diameter;
	//assemble_matrix();
  }

  template<int dim>
  void NavierStokes<dim>::evolve_one_time_step()
  {
	// for the velocity
	assemble_matrix();
	assemble_rhs();
	set_boundary_values(time+dt);
	solve();

	// for phi
	// I assemble the matrix for phi in the run function
	assemble_rhs_phi();
	solve_phi();

	// update the pressure
	// pnp1 = pn + phinp1
	pnp1 = 0.;
	pnp1.add(1.0,pn,1.0,phinp1);
  }

  template<int dim>
  void NavierStokes<dim>::get_boundary_ids()
  {
	boundary_values_id_u.clear();
	boundary_values_id_v.clear();
	
	std::map<unsigned int, double> map_boundary_values_u;
	std::map<unsigned int, double> map_boundary_values_v;
	
	VectorTools::interpolate_boundary_values(dof_handler,
											 0,
											 ExactSolution_u<dim>(time),
											 map_boundary_values_u);
	VectorTools::interpolate_boundary_values(dof_handler,
											 0,
											 ExactSolution_v<dim>(time),
											 map_boundary_values_v);
	std::map<unsigned int,double>::const_iterator iter_u=map_boundary_values_u.begin();
	std::map<unsigned int,double>::const_iterator iter_v=map_boundary_values_v.begin();	
	for (; iter_u !=map_boundary_values_u.end(); ++iter_u)
	  boundary_values_id_u.push_back(iter_u->first);
	for (; iter_v !=map_boundary_values_v.end(); ++iter_v)
	  boundary_values_id_v.push_back(iter_v->first);
  }

  template<int dim>
  void NavierStokes<dim>::set_boundary_values(double time)
  {
	std::vector<double> boundary_values_u;
	std::vector<double> boundary_values_v;
	std::map<unsigned int, double> map_boundary_values_u;
	std::map<unsigned int, double> map_boundary_values_v;
	
	VectorTools::interpolate_boundary_values(dof_handler,
											 0,
											 ExactSolution_u<dim>(time),
											 map_boundary_values_u);
	VectorTools::interpolate_boundary_values(dof_handler,
											 0,
											 ExactSolution_v<dim>(time),
											 map_boundary_values_v);
	std::map<unsigned int,double>::const_iterator iter_u=map_boundary_values_u.begin();
	std::map<unsigned int,double>::const_iterator iter_v=map_boundary_values_v.begin();	
	for (; iter_u !=map_boundary_values_u.end(); ++iter_u)
	  boundary_values_u.push_back(iter_u->second);
	for (; iter_v !=map_boundary_values_v.end(); ++iter_v)
	  boundary_values_v.push_back(iter_v->second);

	// set boundary values to RHS
	system_rhs_u.set(boundary_values_id_u,boundary_values_u);
	system_rhs_u.compress(VectorOperation::insert);	
	system_rhs_v.set(boundary_values_id_v,boundary_values_v);
	system_rhs_v.compress(VectorOperation::insert);
  }

  template <int dim>
  void NavierStokes<dim>::setup_system()
  {
	// FOR THE VELOCITY //
    // LOCALLY OWNED AND LOCALLY RELEVANT DOFs //
    // distributes DoF in parallel
    dof_handler.distribute_dofs(fe); 
    locally_owned_dofs = dof_handler.locally_owned_dofs(); 
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs); // these include ghost cells

    // RHS AND SOLUTION //
	// for u
	unm1.reinit(locally_owned_dofs, mpi_communicator);
	un.reinit(locally_owned_dofs, mpi_communicator);
	unp1.reinit(locally_owned_dofs, mpi_communicator); 
    system_rhs_u.reinit(locally_owned_dofs, mpi_communicator); // the RHS only needs info locally owned
	// for v
	vnm1.reinit(locally_owned_dofs, mpi_communicator);
	vn.reinit(locally_owned_dofs, mpi_communicator);
	vnp1.reinit(locally_owned_dofs, mpi_communicator); 
    system_rhs_v.reinit(locally_owned_dofs, mpi_communicator); // the RHS only needs info locally owned

    // CONSTRAINTS //
    // The next step is to compute constraints like Dirichlet BCs and hanging nodes
	constraints_U.clear();
	constraints_U.reinit(locally_relevant_dofs);
	constraints_U.close();

	// Get Boundary IDs //
	get_boundary_ids();
	  
    // initializing the matrix with sparsity pattern.
    DynamicSparsityPattern dsp_u(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp_u, constraints_U, false); // keep_constrained_dofs=false
    SparsityTools::distribute_sparsity_pattern(dsp_u,
                                               dof_handler.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs);
    system_matrix_u.reinit(locally_owned_dofs,
						   locally_owned_dofs,
						   dsp_u,
						   mpi_communicator);
	DynamicSparsityPattern dsp_v(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp_v, constraints_U, false); // keep_constrained_dofs=false
    SparsityTools::distribute_sparsity_pattern(dsp_v,
                                               dof_handler.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs);
    system_matrix_v.reinit(locally_owned_dofs,
						   locally_owned_dofs,
						   dsp_v,
						   mpi_communicator);

	////// FOR PHI //////
    // LOCALLY OWNED AND LOCALLY RELEVANT DOFs //
    // distributes DoF in parallel
    dof_handler_P.distribute_dofs(fe_P); 
    locally_owned_dofs_P = dof_handler_P.locally_owned_dofs(); 
    DoFTools::extract_locally_relevant_dofs(dof_handler_P, locally_relevant_dofs_P);

    // RHS AND SOLUTION //
	// for phi and p
	phinm1.reinit(locally_owned_dofs_P, mpi_communicator);
	phin.reinit(locally_owned_dofs_P, mpi_communicator);
	phinp1.reinit(locally_owned_dofs_P, mpi_communicator);
    system_rhs_phi.reinit(locally_owned_dofs_P, mpi_communicator);
	pn.reinit(locally_owned_dofs_P, mpi_communicator);
	pnp1.reinit(locally_owned_dofs_P, mpi_communicator); 

    // CONSTRAINTS //
    // The next step is to compute constraints like Dirichlet BCs and hanging nodes
	constraints_phi.clear();
	constraints_phi.reinit(locally_relevant_dofs_P);
	constraints_phi.close();

    // initializing the matrix with sparsity pattern.
    DynamicSparsityPattern dsp_P(locally_relevant_dofs_P);
    DoFTools::make_sparsity_pattern(dof_handler_P, dsp_P, constraints_phi, false);
    SparsityTools::distribute_sparsity_pattern(dsp_P,
                                               dof_handler_P.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs_P);
    system_matrix_phi.reinit(locally_owned_dofs_P,
							 locally_owned_dofs_P,
							 dsp_P,
							 mpi_communicator);
  }

  template <int dim>
  void NavierStokes<dim>::assemble_matrix()
  {
	system_matrix_u = 0.;
	system_matrix_v = 0.;

	// for u
	PETScWrappers::MPI::Vector locally_relevant_un, locally_relevant_unm1;
	locally_relevant_un.reinit(locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
	locally_relevant_unm1.reinit(locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
	locally_relevant_un = un;
	locally_relevant_unm1 = unm1;

	// for v
	PETScWrappers::MPI::Vector locally_relevant_vn, locally_relevant_vnm1;
	locally_relevant_vn.reinit(locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
	locally_relevant_vnm1.reinit(locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
	locally_relevant_vn = vn;
	locally_relevant_vnm1 = vnm1;
	
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
	  
    FullMatrix<double> cell_matrix_u(dofs_per_cell, dofs_per_cell);
	FullMatrix<double> cell_matrix_v(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<double> shape_value(dofs_per_cell);
	std::vector<Tensor<1,dim> > shape_grad(dofs_per_cell);
	std::vector<double> uhn_at_xq(n_q_points);
	std::vector<double> uhnm1_at_xq(n_q_points);
	std::vector<double> vhn_at_xq(n_q_points);
	std::vector<double> vhnm1_at_xq(n_q_points);
	
    // FE loop
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_matrix_u = 0.;
		  cell_matrix_v = 0.;
		  
		  // get shape functions, their derivatives, etc at quad points
		  fe_values.reinit(cell);
		  fe_values.get_function_values(locally_relevant_un,uhn_at_xq);
		  fe_values.get_function_values(locally_relevant_unm1,uhnm1_at_xq);
		  fe_values.get_function_values(locally_relevant_vn,vhn_at_xq);
		  fe_values.get_function_values(locally_relevant_vnm1,vhnm1_at_xq);
		  
		  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
			  // compute detJxW, phi, grad(phi), etc at quad points
			  const double detJxdV = fe_values.JxW(q_point);
			  for (unsigned int i = 0; i < dofs_per_cell; ++i)
				{
				  shape_value[i] = fe_values.shape_value(i,q_point);
				  shape_grad[i]  = fe_values.shape_grad(i,q_point);
				}

			  double uhStar = 2*uhn_at_xq[q_point]-uhnm1_at_xq[q_point];
			  double vhStar = 2*vhn_at_xq[q_point]-vhnm1_at_xq[q_point];
			  
			  // loop on i-DoFs
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
				  // loop on j-DoFs
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
					{
					  cell_matrix_u(i,j) += (shape_value[i] * shape_value[j] 
											 +
											 2*dt/3.0*(uhStar*shape_grad[j][0]
													   +vhStar*shape_grad[j][1])*shape_value[i]
											 +
											 2*dt*coeff_mu/3.0/coeff_rho * (shape_grad[i] * shape_grad[j])
											 ) * detJxdV;
					  cell_matrix_v(i,j) += (shape_value[i] * shape_value[j] 
											 +
											 2*dt/3.0*(uhStar*shape_grad[j][0]
													   +vhStar*shape_grad[j][1])*shape_value[i]
											 +
											 2*dt*coeff_mu/3.0/coeff_rho * (shape_grad[i] * shape_grad[j])
											 ) * detJxdV;
					}
                }
            }
		  // assemble from local to global operators
          cell->get_dof_indices(local_dof_indices);
          constraints_U.distribute_local_to_global(cell_matrix_u,
												   local_dof_indices,
												   system_matrix_u);
          constraints_U.distribute_local_to_global(cell_matrix_v,
												   local_dof_indices,
												   system_matrix_v);		  
        }
    // Distribute between processors
    system_matrix_u.compress(VectorOperation::add);
	system_matrix_v.compress(VectorOperation::add);

	// clear rows related to BCs
	system_matrix_u.clear_rows(boundary_values_id_u,1.0);
	system_matrix_v.clear_rows(boundary_values_id_v,1.0);
							  
	// compute preconditioners
	preconditioner_u=NULL;
	preconditioner_u.reset(new PETScWrappers::PreconditionBlockJacobi(system_matrix_u));
	preconditioner_v=NULL;
	preconditioner_v.reset(new PETScWrappers::PreconditionBlockJacobi(system_matrix_v));
  }

  template <int dim>
  void NavierStokes<dim>::assemble_rhs()
  {
	system_rhs_u = 0.;
	system_rhs_v = 0.;

	// for u
	PETScWrappers::MPI::Vector locally_relevant_un, locally_relevant_unm1;
	locally_relevant_un.reinit(locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
	locally_relevant_unm1.reinit(locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
	locally_relevant_un = un;
	locally_relevant_unm1 = unm1;

	// for v
	PETScWrappers::MPI::Vector locally_relevant_vn, locally_relevant_vnm1;
	locally_relevant_vn.reinit(locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
	locally_relevant_vnm1.reinit(locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
	locally_relevant_vn = vn;
	locally_relevant_vnm1 = vnm1;
	
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
	
	Vector<double>     cell_rhs_u(dofs_per_cell);
	Vector<double>     cell_rhs_v(dofs_per_cell);
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	std::vector<double> uhn_at_xq(n_q_points);
	std::vector<double> uhnm1_at_xq(n_q_points);
	std::vector<double> vhn_at_xq(n_q_points);
	std::vector<double> vhnm1_at_xq(n_q_points);
	
	FunctionRHS_u<dim> function_u;
	function_u.set_time(time+dt);
	FunctionRHS_v<dim> function_v;
	function_v.set_time(time+dt);
	
	// FE loop
	for (const auto &cell : dof_handler.active_cell_iterators())
	  if (cell->is_locally_owned())
		{
		  cell_rhs_u = 0.;
		  cell_rhs_v = 0.;
		  
		  // get shape functions, their derivatives, etc at quad points
		  fe_values.reinit(cell);
		  fe_values.get_function_values(locally_relevant_un,uhn_at_xq);
		  fe_values.get_function_values(locally_relevant_unm1,uhnm1_at_xq);
		  fe_values.get_function_values(locally_relevant_vn,vhn_at_xq);
		  fe_values.get_function_values(locally_relevant_vnm1,vhnm1_at_xq);
		  
		  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
			{
			  // compute detJxW, phi, grad(phi), etc at quad points
			  const double detJxdV = fe_values.JxW(q_point);
			  
			  const double rhs_value_u
				= 4.0/3.0 * uhn_at_xq[q_point] - 1.0/3.0 * uhnm1_at_xq[q_point] +
				2*dt/3.0/coeff_rho * function_u.value(fe_values.quadrature_point(q_point));

			  const double rhs_value_v
				= 4.0/3.0 * vhn_at_xq[q_point] - 1.0/3.0 * vhnm1_at_xq[q_point] +
				2*dt/3.0/coeff_rho * function_v.value(fe_values.quadrature_point(q_point));
			  
			  // loop on i-DoFs
			  for (unsigned int i = 0; i < dofs_per_cell; ++i)
				{
				  cell_rhs_u(i) += rhs_value_u * fe_values.shape_value(i,q_point) * detJxdV;
				  cell_rhs_v(i) += rhs_value_v * fe_values.shape_value(i,q_point) * detJxdV;
				}
			}
		  
		  // assemble from local to global operators
		  cell->get_dof_indices(local_dof_indices);
		  constraints_U.distribute_local_to_global(cell_rhs_u,
												   local_dof_indices,
												   system_rhs_u);
		  constraints_U.distribute_local_to_global(cell_rhs_v,
												   local_dof_indices,
												   system_rhs_v);		  
		}
	// Distribute between processors
	system_rhs_u.compress(VectorOperation::add);
	system_rhs_v.compress(VectorOperation::add);
  }
	
  template <int dim>
  void NavierStokes<dim>::solve()
  {
    PETScWrappers::MPI::Vector solution_u(locally_owned_dofs,mpi_communicator);
	PETScWrappers::MPI::Vector solution_v(locally_owned_dofs,mpi_communicator);
	solution_u = 0.;
	solution_v = 0.;
    SolverControl solver_control(dof_handler.n_dofs(), 1e-12);
	use_iterative_solver=true;
    if (use_iterative_solver)
      {
		PETScWrappers::SolverCG solver(solver_control, mpi_communicator);
		solver.solve(system_matrix_u,
					 solution_u,
					 system_rhs_u,
					 *preconditioner_u);
		solver.solve(system_matrix_v,
					 solution_v,
					 system_rhs_v,
					 *preconditioner_v);
      }
    else
      {
		PETScWrappers::SparseDirectMUMPS solver(solver_control,mpi_communicator);
		solver.solve(system_matrix_u,
					 solution_u,
					 system_rhs_u);
		solver.solve(system_matrix_v,
					 solution_v,
					 system_rhs_v);		
      }
	if (verbosity==1)
	  pcout << "   Solved in " << solver_control.last_step() << " iterations." << std::endl;

    // apply constraints
    constraints_U.distribute(solution_u);
    unp1 = solution_u;

	constraints_U.distribute(solution_v);
    vnp1 = solution_v;
  }

  template <int dim>
  void NavierStokes<dim>::output_results(const unsigned int cycle) const
  {
	PETScWrappers::MPI::Vector locally_relevant_unp1, locally_relevant_vnp1;
	locally_relevant_unp1.reinit(locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
	locally_relevant_vnp1.reinit(locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
		
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    std::vector<std::string> solution_names(1);

	// for u
    solution_names[0] = "u";
	locally_relevant_unp1 = unp1;
    data_out.add_data_vector(locally_relevant_unp1, solution_names);

	// for v
	solution_names[0] = "v";
	locally_relevant_vnp1 = vnp1;
    data_out.add_data_vector(locally_relevant_vnp1, solution_names);

    data_out.build_patches(mapping,degree,DataOut<dim>::no_curved_cells);
    data_out.write_vtu_with_pvtu_record("./", "solution", cycle, mpi_communicator, 2, 8);
  }
  
  template <int dim>
  void NavierStokes<dim>::get_errors(const unsigned int cycle)
  {
	// for u
	PETScWrappers::MPI::Vector locally_relevant_unp1;
    locally_relevant_unp1.reinit(locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
	locally_relevant_unp1 = unp1;
    Vector<double> difference_per_cell_u(triangulation.n_active_cells());
    VectorTools::integrate_difference(mapping,
									  dof_handler,
									  locally_relevant_unp1,
									  ExactSolution_u<dim>(time),
									  difference_per_cell_u,
									  QGauss<dim>(fe.degree + 1),
									  VectorTools::L2_norm);
    const double L2_error_u =
      VectorTools::compute_global_error(triangulation,
										difference_per_cell_u,
										VectorTools::L2_norm);
	// for v
	PETScWrappers::MPI::Vector locally_relevant_vnp1;
    locally_relevant_vnp1.reinit(locally_owned_dofs,locally_relevant_dofs,mpi_communicator);
	locally_relevant_vnp1 = vnp1;
    Vector<double> difference_per_cell_v(triangulation.n_active_cells());
    VectorTools::integrate_difference(mapping,
									  dof_handler,
									  locally_relevant_vnp1,
									  ExactSolution_v<dim>(time),
									  difference_per_cell_v,
									  QGauss<dim>(fe.degree + 1),
									  VectorTools::L2_norm);
    const double L2_error_v =
      VectorTools::compute_global_error(triangulation,
										difference_per_cell_v,
										VectorTools::L2_norm);
	
    pcout << "   L2 error for cycle "
		  << cycle
		  << ": "
		  << L2_error_u
		  << ", "
		  << L2_error_v
		  << std::endl;

    // save error into convergence_table
    const unsigned int n_active_cells=triangulation.n_global_active_cells();
    const unsigned int n_dofs = dof_handler.n_dofs();	
    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("n_dofs", n_dofs);
    convergence_table.add_value("E2_u", L2_error_u);
	convergence_table.add_value("E2_v", L2_error_v);
  }
  
  template <int dim>
  void NavierStokes<dim>::run()
  {
    pcout << "Running with "
          << "PETSc"
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

	coeff_rho = 1.0;
	coeff_mu = 1.0;
	
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

	unsigned int num_cycles = 3;
	for (unsigned int cycle=0; cycle<num_cycles; ++cycle)
	  {
		// ***** Initial time ***** //
		time = 0.0;
		
		// ***** DOMAIN ***** /
		if (cycle==0)
		  {
			unsigned int refinement = 4; // initial refinement
			GridGenerator::hyper_cube(triangulation);
			triangulation.refine_global(refinement);
		  }
		else
		  {
			triangulation.refine_global(1);
			//dt = output_time/int(output_time/(cfl*min_cell_diameter));
		  }
		
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
		//dt = cfl*min_cell_diameter;
		dt = output_time/int(output_time/(cfl*min_cell_diameter));
		
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
		get_errors(cycle);
		if (cycle==0)
		  {
			convergence_table.set_precision("E2_u", 2);
			convergence_table.set_scientific("E2_u",true);
			convergence_table.set_precision("E2_v", 2);
			convergence_table.set_scientific("E2_v",true);			
			convergence_table.set_tex_format("cells","r");
			convergence_table.set_tex_format("n_dofs","r");
		  }		
	  } // cycles

	// ***** GET CONVERGENCE RATES AND PRINT TABLE ***** //
    convergence_table.evaluate_convergence_rates("E2_u", "cells", ConvergenceTable::reduction_rate_log2, dim);
	convergence_table.evaluate_convergence_rates("E2_v", "cells", ConvergenceTable::reduction_rate_log2, dim);
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
		std::cout << std::endl;
		convergence_table.write_text(std::cout);
      }
  } //run
}  //
	
int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace NS;
      
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      
      unsigned int degree = 1;
      NavierStokes<2> ns_problem_2d(degree);
      ns_problem_2d.run();
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

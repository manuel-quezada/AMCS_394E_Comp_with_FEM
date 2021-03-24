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

namespace GeoCoord
{
  using namespace dealii::GeometricUtilities::Coordinates;
} 

namespace Poisson
{
  using namespace dealii;

  template <int dim>
  class ExactSolution : public Function <dim>
  {
  public:
	ExactSolution(double omega) : Function<dim>() {this->omega=omega;}
	virtual double value (const Point<dim> &p, const unsigned int component=0) const;
	double omega=omega;
  };
  
  template <int dim>
  double ExactSolution<dim>::value (const Point<dim> &p, const unsigned int) const
  {
	std::array<double, dim> polar = GeoCoord::to_spherical(p);
	double r = polar[0];
	double theta = polar[1];	
	return std::sin(M_PI*r)*std::cos(omega*theta);
  }

  template <int dim>
  class ForceTerm : public Function <dim>
  {
  public:
	ForceTerm(double omega) : Function<dim>() {this->omega=omega;}
	virtual double value (const Point<dim> &p, const unsigned int component=0) const;
	double omega;
  };
  
  template <int dim>
  double ForceTerm<dim>::value (const Point<dim> &p, const unsigned int) const
  {
	std::array<double, dim> polar = GeoCoord::to_spherical(p);
	double r = polar[0];
	double th = polar[1];
	double pir = M_PI*r;

	return -1.0/r/r * (pir*std::cos(pir)
					   -(pir*pir + omega*omega)*std::sin(pir)
					   )*std::cos(omega*th);
  }

  template <int dim>
  class RadialNeumannBC : public Function <dim>
  {
  public:
	RadialNeumannBC(double omega, double R) : Function<dim>() {this->omega=omega; this->R=R;}
	virtual double value (const Point<dim> &p, const unsigned int component=0) const;
	double omega;
	double R;
  };
  
  template <int dim>
  double RadialNeumannBC<dim>::value (const Point<dim> &p, const unsigned int) const
  {
	std::array<double, dim> polar = GeoCoord::to_spherical(p);
	double th = polar[1];
	double piR = M_PI*R;

	return M_PI*std::cos(piR)*std::cos(omega*th);
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

    parallel::distributed::Triangulation<dim> triangulation;

	// numerics
	double degree;
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
	double inner_radius,outer_radius;
	double omega;
	
	// boundary conditions
	bool outer_dirichlet_boundary;
	unsigned int inner_boundary_id;
	unsigned int outer_boundary_id;
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
    dof_handler.distribute_dofs(fe); // distributes DoF in parallel
    locally_owned_dofs = dof_handler.locally_owned_dofs(); 
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs); // these include ghost cells

	// RHS AND SOLUTION //
    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator); // the RHS only needs info locally owned

	// CONSTRAINTS //
    // The next step is to compute dirichlet boundary value constraints
	// Note: we don't need high-order mappings to impose Dirichlet BCs strongly
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
	VectorTools::interpolate_boundary_values(mapping,
											 dof_handler,
											 inner_boundary_id,
											 ExactSolution<dim>(omega),
											 constraints);
	if (outer_dirichlet_boundary)
	  VectorTools::interpolate_boundary_values(mapping,
											   dof_handler,
											   outer_boundary_id,
											   ExactSolution<dim>(omega),
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
  }

  template <int dim>
  void PoissonProblem<dim>::assemble_system()
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);
	const QGauss<dim-1> face_quadrature_formula(fe.degree + 1);
	
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
									 update_values |
									 update_quadrature_points |
									 update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();
	const unsigned int n_face_q_points = face_quadrature_formula.size();
	  
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	std::vector<Tensor<1, dim> > shape_grad(dofs_per_cell);
	std::vector<double> shape_value(dofs_per_cell);
	
	ForceTerm<dim> force(omega);
	RadialNeumannBC<dim> radial_neumann_bc(omega,outer_radius);
	
	// loop on cells
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_matrix = 0.;
          cell_rhs    = 0.;

		  fe_values.reinit(cell); // get shape functions and their derivatives at quad points

		  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
			  // compute detJxW, phi and grad(phi) at quad points
			  const double detJxdV = fe_values.JxW(q_point);
			  for (unsigned int i = 0; i < dofs_per_cell; ++i)
				{
				  shape_grad[i] = fe_values.shape_grad(i,q_point);
				  shape_value[i] = fe_values.shape_value(i,q_point);
				}

			  const double rhs_value = force.value(fe_values.quadrature_point(q_point));
			  // loop on i-DoFs
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
				  cell_rhs(i) += rhs_value * shape_value[i] * detJxdV;
				  // loop on j-DoFs
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
					cell_matrix(i, j) += (shape_grad[i] * shape_grad[j]) * detJxdV;
                }
            }

		  // ***** loop on faces ***** //
		  if (outer_dirichlet_boundary == false)
			for (unsigned int face_no : GeometryInfo<dim>::face_indices())
			  {
				if (cell->face(face_no)->at_boundary() &&
					(cell->face(face_no)->boundary_id()==outer_boundary_id))
				  {
					fe_face_values.reinit(cell, face_no);
					for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point)
					  {
						double detJxdV = fe_face_values.JxW(q_point);
						const double rhs_value = radial_neumann_bc.value(fe_face_values.quadrature_point(q_point));
						// loop on i- and j-DoFs
						for (unsigned int i = 0; i < dofs_per_cell; ++i)
						  cell_rhs(i) += rhs_value * fe_face_values.shape_value(i, q_point) * detJxdV;
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

    //PETScWrappers::SolverCG solver(solver_control, mpi_communicator);
	//PETScWrappers::SolverBicgstab solver(solver_control,mpi_communicator);
	//PETScWrappers::SolverGMRES solver(solver_control, mpi_communicator);
	//PETScWrappers::SolverLSQR solver(solver_control, mpi_communicator);
	//PETScWrappers::SolverChebychev solver(solver_control, mpi_communicator);
    //PETScWrappers::PreconditionBoomerAMG preconditioner;
    //PETScWrappers::PreconditionBoomerAMG::AdditionalData data;

    //data.symmetric_operator = true;
    //preconditioner.initialize(system_matrix, data);
    //solver.solve(system_matrix,
	//         completely_distributed_solution,
	//         system_rhs,
	//			 preconditioner);

	PETScWrappers::SparseDirectMUMPS solver(solver_control,mpi_communicator);
	solver.solve(system_matrix,
				 completely_distributed_solution,
				 system_rhs);

    pcout << "   Solved in " << solver_control.last_step() << " iterations." << std::endl;

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
    data_out.write_vtu_with_pvtu_record(
      "./", "solution", cycle, mpi_communicator, 2, 8);
  }
  
  template <int dim>
  void PoissonProblem<dim>::get_errors(const unsigned int cycle)
  {
	Vector<double> difference_per_cell(triangulation.n_active_cells());
	VectorTools::integrate_difference(mapping,
									  dof_handler,
									  locally_relevant_solution,
									  ExactSolution<dim>(omega),
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
	// boundary
	inner_radius = 0.25;
	outer_radius = 1.0;
	omega = 4.0;
	outer_dirichlet_boundary = false;
	
	// ******************************** //
	// ***** numerical parameters ***** //
	// ******************************** //
	unsigned int refinement = 4; // initial refinement
	bool get_error=true;
	unsigned int num_cycles = 5;
	
	for (unsigned int cycle = 0; cycle < num_cycles; ++cycle)
	  {
		pcout << "Cycle " << cycle << ':' << std::endl;
		if (cycle == 0)
		  {
			// ***** domain ***** //
			inner_boundary_id = 0;
			outer_boundary_id = 1;
			Point<dim> center(0.,0.);
			int n_cells = 0;
			GridGenerator::hyper_shell(triangulation,
									   center,
									   inner_radius,
									   outer_radius,
									   n_cells = n_cells,
									   true);
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

	  unsigned int degree = 2; 
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

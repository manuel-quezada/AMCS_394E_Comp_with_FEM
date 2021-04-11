  template <int dim>
  class HeatEquation
  {
  public:
    HeatEquation(const unsigned int degree);

    void run();

  private:
    void get_initial_condition();
    void setup_system();
    void evolve_to_time(const double final_time);
    void evolve_one_time_step();

    // time related quantities
    double time;
    double final_time;
    double dt;
    double cfl;
    double min_cell_diameter;
    unsigned int verbosity;
  };

  // constructor 
  template <int dim>
  HeatEquation<dim>::HeatEquation(const unsigned int degree) {}

  template<int dim>
  void HeatEquation<dim>::get_initial_condition()
  {
    TimerOutput::Scope t(computing_timer, "get initial condition");
    PETScWrappers::MPI::Vector completely_distributed_init_condition(locally_owned_dofs, mpi_communicator);
    VectorTools::interpolate(mapping,
			     dof_handler, 
			     InitCond<dim>(), 
			     completely_distributed_init_condition);
    locally_relevant_un = completely_distributed_init_condition;
    unp1 = completely_distributed_init_condition; // to output the init condition
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
	locally_relevant_un = unp1;

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
	      }
	  }
      }
    // adjust dt 
    dt = cfl*min_cell_diameter;
  }

  template<int dim>
  void HeatEquation<dim>::evolve_one_time_step()
  {
    // assemble rhs, solve the system (M+dt*S)*U^{n+1} = R^{n+1}, update U^n=U^{n+1} for next time step
  }
  
  template <int dim>
  void HeatEquation<dim>::run()
  {
    pcout << "Running with "
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    // ***** OUTPUT CONTROLLER ***** //
    // Number of output files
    unsigned int nOut = int(final_time/output_time)+1;
    nOut += (nOut-1)*output_time < final_time ? 1 : 0;
    // create the tnList
    std::vector<double> tnList(nOut);
    for (unsigned int i=0; i<nOut; ++i)
      tnList[i] = i*output_time <= final_time ? i*output_time : final_time; 

    // ***** Initial time ***** //
    time = 0.0; 

    // ***** DOMAIN ***** //
    // ...

    // ***** SETUP ***** //
    // ...
	
    // ***** INITIAL CONDITIONS ***** //
    get_initial_condition();
	
    // output initial condition //
    {
      TimerOutput::Scope t(computing_timer, "output");
      output_results(0);
    }
	
    // ***** ASSEMBLE MASS MATRIX ***** //
    // ..
	
    // more numerical parameters
    min_cell_diameter = GridTools::minimal_cell_diameter(triangulation);
    cfl=0.5;
    dt = cfl*min_cell_diameter;
    
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
	  TimerOutput::Scope t(computing_timer, "output");
	  output_results(time_interval);
	}
      } // end of time_interval
    
    // ***** print results of timer ***** //
  } // run
} // namespace heat_equation

int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace heat_equation;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      unsigned int degree = 1;
      HeatEquation<2> heat_equation_2d(degree);
      heat_equation_2d.run();
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

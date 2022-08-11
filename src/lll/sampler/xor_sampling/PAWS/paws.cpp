/** PAWS: PArity-based Weighted Sampler */

#include "paws.h"

unsigned long seed;
bool use_given_seed = false;
IloInt timelimit = -1;
char instanceName[1024];
char outpath[1024];

/**********************/

void parseArgs(int argc, char **argv)
{
  // one argument must be the instance filename
  if (argc <= 1)
  {
    cerr << "ERROR: instance name must be specified" << endl
         << "       See usage (paws -h)" << endl;
    exit(1);
  }

  // default toulbar verbosity level
  ToulBar2::verbose = 0;

  for (int argIndex = 1; argIndex < argc; ++argIndex)
  {

    if (!strcmp(argv[argIndex], "-timelimit"))
    {
      argIndex++;
      timelimit = atol(argv[argIndex]);
    }
    else if (!strcmp(argv[argIndex], "-seed"))
    {
      argIndex++;
      seed = atol(argv[argIndex]);
      use_given_seed = true;
    }
    else if (!strcmp(argv[argIndex], "-samples"))
    {
      argIndex++;
      samples_num = atol(argv[argIndex]);
    }
    else if (!strcmp(argv[argIndex], "-nbauxv"))
    {
      argIndex++;
      nbauxvars = atol(argv[argIndex]);
    }
    else if (!strcmp(argv[argIndex], "-alpha"))
    {
      argIndex++;
      alpha = atol(argv[argIndex]);
    }
    else if (!strcmp(argv[argIndex], "-merge"))
    {
      merge_singleton_factors = true;
    }
    else if (!strcmp(argv[argIndex], "-burningIn"))
    {
      argIndex++;
      burningIn = atol(argv[argIndex]);
    }
    else if (!strcmp(argv[argIndex], "-pivot"))
    {
      argIndex++;
      pivot = atol(argv[argIndex]);
    }
    else if (!strcmp(argv[argIndex], "-b"))
    {
      argIndex++;
      b = atol(argv[argIndex]);
    }
    else if (!strcmp(argv[argIndex], "-verbosity"))
    {
      argIndex++;
      ToulBar2::verbose = atoi(argv[argIndex]);
    }
    else if (!strcmp(argv[argIndex], "-outpath"))
    {
      argIndex++;
      strcpy(outpath, argv[argIndex]);
    }
    else if (!strcmp(argv[argIndex], "-h") || !strcmp(argv[argIndex], "-help"))
    {
      cout << endl
           << "USAGE: paws [options] instance.uai" << endl
           << endl
           << "   -timelimit          Timelimit in seconds (default None)" << endl
           << "   -seed               Random seed" << endl
           << "   -samples            #samples" << endl
           << "   -alpha              alpha" << endl
           << "   -b            	  b" << endl
           << "   -pivot              pivot" << endl
           << "   -nbauxv             number of auxiliary vars" << endl
           << endl;
      // print parity constraint options usage
      printParityUsage(cout);
      exit(0);
    }
    else if (argv[argIndex][0] != '-')
    {
      static bool instanceNameSpecified = false;
      if (!instanceNameSpecified)
      {
        // must be the instance name
        strcpy(instanceName, argv[argIndex]);
        instanceNameSpecified = true;
      }
      else
      {
        cerr << "ERROR: incorrect number of non-option arguments." << endl
             << "       See usage (paws -h)" << endl;
        exit(1);
      }
    }
    else
    {
      cerr << "ERROR: Unexpected option: " << argv[argIndex] << endl
           << "       See usage (paws -h)" << endl;
      exit(1);
    }
  }
}

void compute_marginals(std::vector<std::vector<size_t>> final_samples)
{
  if (final_samples.empty())
    return;

  size_t nvar = final_samples[0].size();
  size_t nsamples = final_samples.size();

  std::vector<std::vector<double>> marginals;
  marginals.resize(nvar);
  for (size_t j = 0; j < nvar; j++)
    marginals[j].resize(2);

  for (size_t i = 0; i < final_samples.size(); i++)
  {
    for (size_t j = 0; j < nvar; j++)
      marginals[j][final_samples[i][j]]++;
  }

  cout << "Computing marginals using " << nsamples << " samples" << endl;
  for (size_t j = 0; j < nvar; j++)
    cout << "var" << j << "," << marginals[j][0] / nsamples << "," << marginals[j][1] / nsamples << endl;
}

//////////////////////

void compute_marginals_weighted(std::vector<std::pair<std::vector<size_t>, double>> final_samples)
{
  if (final_samples.empty())
    return;

  size_t nvar = final_samples[0].first.size();
  // size_t nsamples = final_samples.size();

  std::vector<std::vector<double>> marginals;
  marginals.resize(nvar);
  for (size_t j = 0; j < nvar; j++)
    marginals[j].resize(2);

  double tot = 0.0;
  for (size_t i = 0; i < final_samples.size(); i++)
  {
    for (size_t j = 0; j < nvar; j++)
    {
      marginals[j][final_samples[i].first[j]] += final_samples[i].second;
    }
    tot += final_samples[i].second;
  }

  cout << "Computing marginals using " << tot << " samples" << endl;
  for (size_t j = 0; j < nvar; j++)
    cout << "var" << j << "," << marginals[j][0] / tot << "," << marginals[j][1] / tot << endl;
}

IlcWeightedCSPI::IlcWeightedCSPI(IloCP solver,
                                 IlcIntVar objective, IlcIntVarArray variables,
                                 const char *fileName) : IlcConstraintI(solver), obj(objective),
                                                         size(variables.getSize()), vars(variables), wcsp(NULL),
                                                         unassignedVars(NULL), currentNumberOfFails(0),
                                                         synchronized(solver, IlcTrue)
{
  // creates a WCSP object
  wcsp = WeightedCSP::makeWeightedCSP(&STORE, MAX_COST);
  // load WCSP problem from a file if available
  if (fileName)
  {
    // stefano mod
    wcsp->read_uai2008(fileName);
    wcspmarkov_log = ToulBar2::markov_log;
    wcsplb = wcsp->getNegativeLb();
    wcspnf = ToulBar2::NormFactor;
    ToulBar2::showSolutions = true;
    //	ToulBar2::verbose = 2;
    // wcsp->read_wcsp(fileName);
    assert((unsigned int)size == wcsp->numberOfVariables());
  }
  // specific data to check if all variables have been assigned
  unassignedVars = new Domain(0, size - 1, &STORE.storeDomain);
  // memorizes all WeightedCSP instances
  assert(wcsp->getIndex() == wcspCounter);
  AllIlcWeightedCSPI.push_back(this);
  wcspCounter++;
  CurrentWeightedCSP = wcsp;
}

void IlcWeightedCSPI::synchronize()
{
  if (ToulBar2::verbose >= 2)
    cout << "Domain synchronization between IlogSolver and Toulbar2!" << endl;
  for (int i = 0; i < size; i++)
  {
    if (ToulBar2::verbose >= 2)
      cout << vars[i] << " (" << wcsp->getInf(i) << "," << wcsp->getSup(i) << ")" << endl;
    vars[i].setMin(wcsp->getInf(i));
    vars[i].setMax(wcsp->getSup(i));
    for (int d = wcsp->getInf(i); d <= wcsp->getSup(i); d++)
    {
      if (wcsp->cannotbe(i, d))
      {
        vars[i].removeValue(d);
      }
    }
    wcsp->increase(i, vars[i].getMin());
    wcsp->decrease(i, vars[i].getMax());
    for (int d = vars[i].getMin(); d <= vars[i].getMax(); d++)
    {
      if (!vars[i].isInDomain(d))
      {
        wcsp->remove(i, d);
      }
    }
  }
  cout << "synch: lower bound" << wcsp->getLb() << " upperb=" << wcsp->getUb() << endl;

  obj.setMin(wcsp->getLb());
  obj.setMax(wcsp->getUb() - 1);
  wcsp->decreaseUb(obj.getMax() + 1);
  UpperBound = wcsp->getUb();
  cout << "UpperBound=" << UpperBound << ", wcspub=" << wcsp->getUb() << endl;
}

//////////////////////
//////////////////////

// Usage: paws problem_name.wcsp [verbosity]
int main(int argc, char **argv)
{

  string pbname;
  int nbvar, nbval, nbconstr;
  IloEnv env;
  IloTimer timer(env);
  strcpy(outpath, "xor_res.txt");

  try
  {
    // first parse and remove parity-related command-line arguments
    PARITY_DONT_HANDLE_RANDOM_SEED = true;
    parseParityArgs(argc, argv);
    // now parse regular arguments
    parseArgs(argc, argv);

    IloModel model(env);

    // open the instance file
    cout << "open instance: " << instanceName << endl;
    ifstream file(instanceName);
    if (!file)
    {
      cerr << "Could not open file " << instanceName << endl;
      exit(EXIT_FAILURE);
    }

    // read uai file
    // reads uai file to parse domain sizes; creates variables along the way
    cerr << "Creating variables" << endl;
    file >> pbname;
    file >> nbvar;
    IloIntVarArray vars(env, nbvar, 0, MAX_DOMAIN_SIZE - 1);
    nbval = 0;
    int tmp;
    for (int i = 0; i < nbvar; i++)
    {
      file >> tmp;
      if (tmp > nbval)
        nbval = tmp;
      vars[i].setBounds(0, tmp - 1);
      char *name = new char[16];
      sprintf(name, "x%d", i);
      vars[i].setName(name);
    }
    model.add(vars);

    file >> nbconstr;
    cerr << "Var:" << nbvar << " max dom size:" << nbval << " constraints:" << nbconstr << endl;

    // creates the objective function
    IloIntVar obj(env, 0, MAX_COST, "objective");

    // creates a global weighted CSP constraint
    model.add(IloWeightedCSP(env, obj, vars, instanceName));

    IloCP solver(env);
    solver.extract(model);
    IlogSolver = solver;

    // set CP Optimizer's parameters
    if (timelimit > 0)
      solver.setParameter(IloCP::TimeLimit, timelimit);
    solver.setParameter(IloCP::LogPeriod, 10000);
    solver.setParameter(IloCP::Workers, 1); // number of parallel threads

    Objective = solver.getIntVar(obj);
    ProblemSize = nbvar;
    ProblemVars = solver.getIntVarArray(vars);
    BestSol = new int[nbvar];
    BestSol[0] = -1;

    timer.start();

    solver.solve(IloGenerateVars(env, vars) && IloNewSolution(env));
    STORE.restore(0);

    if (BestSol[0] == -1)
      cout << "Proved Infeasibility in " << solver.getInfo(IloCP::NumberOfFails) << " fails and " << solver.getTime() << " seconds." << endl;
    else
    {
      cout << "Optimum: " << UpperBound << " log10like: " << -(UpperBound + wcsplb) / wcspnf + wcspmarkov_log << " in " << solver.getInfo(IloCP::NumberOfFails) << " fails and " << solver.getTime() << " seconds." << endl;
      if (ToulBar2::verbose >= 0)
      {
        cout << "Optimal solution: ";
        for (int i = 0; i < nbvar; i++)
        {
          cout << BestSol[i];
        }
        cout << endl;
      }
    }
    solver.printInformation();

    ////////////////////////////
    cout << "wcsplb=" << wcsplb << endl;
    cout << "wcspnf=" << wcspnf << endl;
    cout << "wcspmarkovlog=" << wcspmarkov_log << endl;

    normfCPO = 1.0 / wcspnf;
    M0 = UpperBound; // solver.getValue(obj);
    approx_factor = pow(2.0, b) / (pow(2.0, b) - 1.0);

    // add auxiliary Y variables
    if (nbauxvars < 0)
      nbauxvars = nbvar;

    int nbuckets = nbauxvars;

    // b bits per bucket
    nbauxvars = nbauxvars * b;

    IloBoolVarArray Yvars(env, nbauxvars);
    nbval = 0;

    for (int i = 0; i < nbuckets; i++)
    {
      char *name = new char[16];
      sprintf(name, "y%d", i);
      Yvars[i].setName(name);
      cout << "y" << i << ": " << ceil(M0 + (i + 1) * log10(approx_factor) / normfCPO) << "," << ceil(M0 + i * log10(approx_factor) / normfCPO) << endl;

      IloOr myor(env);
      for (int kk = 0; kk < b; kk++)
        myor.add(Yvars[i * b + kk] == 1);
      model.add(IloIfThen(env, (obj >= (IloInt)ceil(M0 + (i + 1) * log10(approx_factor) / normfCPO)), myor));
    }

    model.add(Yvars);

    cout << "obj <= " << ceil(M0 + (nbuckets)*log10(approx_factor) / normfCPO) << endl;
    model.add((IloInt)ceil(M0 + (nbuckets)*log10(approx_factor) / normfCPO) >= obj);

    Mmax = (IloInt)ceil(M0 + (nbauxvars)*log10(approx_factor) / normfCPO);

    IloIntVarArray active_vars(env, 0);
    active_vars.add(vars);
    active_vars.add(Yvars);

    std::vector<size_t> current_sample;
    current_sample.resize(nbvar);

    std::vector<std::vector<size_t>> final_samples;
    std::vector<std::pair<std::vector<size_t>, double>> final_samples_weighted;
    std::vector<std::vector<size_t>> total_final_samples;

    int level = 0;

    for (unsigned sample = 0; sample <= samples_num; sample++)
    {
      int nbr_diff_solution_found = 0;

      if (sample < burningIn)
        level = 0;

      ///////////////////////////////////////////////////////////////////////////////
      // keep increasing the number of XORS until enumeration is (typically) feasible
      ////////////////////////////////////////////////////////////////////////////////
      while (0 == 0)
      {
        int largercount = 0;
        int smallercount = 0;
        int T;
        if (sample == 0)
          T = 5;
        else
          T = 1;

        for (int t = 0; t < T; t++)
        {
          // add level xors
          parity_number = level;
          cout << "----------------------------------------" << endl;
          cout << "Sample " << sample << ", level:" << level << endl;
          cout << "----------------------------------------" << endl;

          IloConstraint par(IloParityConstraint1(env, active_vars));
          model.add(par);

          // set CP Optimizer's parameters
          if (timelimit > 0)
            solver.setParameter(IloCP::TimeLimit, timelimit);
          solver.setParameter(IloCP::LogPeriod, 1000);
          solver.setParameter(IloCP::Workers, 6); // number of parallel threads
          solver.setParameter(IloCP::LogVerbosity, IloCP::Quiet);

          solver.extract(model);
          IlogSolver = solver;
          timer.start();

          samples_pool.clear();
          nbr_diff_solution_found = 0;
          IloConstraintArray solutions_removed(env);

          Objective = solver.getIntVar(obj);
          ProblemSize = nbvar;
          ProblemVars = solver.getIntVarArray(vars);
          BestSol = new int[nbvar];
          BestSol[0] = -1;

          AuxiliaryVars = solver.getIntVarArray(Yvars);

          lastConflictVar = -1;
          optimize_sol = false;

          // IloParallelSolver psolver(model, 3);
          // psolver.solve(IloGenerateVars(env,vars) && IloGenerate(env,Yvars)&& IloNewSolution(env));

          // solver.solve(IloGenerateVars(env,vars) && IloNewSolution(env));
          solver.solve(IloGenerateVars(env, vars) && IloGenerate(env, Yvars) && IloNewSolution(env));

          // solver.startNewSearch(IloGenerateVars(env,vars) && IloGenerate(env,Yvars)&& IloNewSolution(env));
          // solver.next();

          STORE.restore(0);

          // solver.endSearch();
          model.remove(par);
          nbr_diff_solution_found = samples_pool.size();
          cout << "Found " << nbr_diff_solution_found << " at level " << level << endl;

          if (nbr_diff_solution_found >= (int)pivot)
            largercount++;
          else
            smallercount++;

          // cout << "largercount " << largercount << " smallercount " << smallercount << endl;

          if (2 * largercount >= T)
            break;
          if (2 * smallercount >= T)
            break;
        }

        if (sample == 0)
        {
          if (2 * largercount >= T) // consistntly larger, add more xors
            level++;
          else
          {
            level = level + alpha;
            break;
          }
        }
        else
          break;
      }

      for (std::set<std::vector<size_t>>::iterator it = samples_pool.begin(); it != samples_pool.end(); ++it)
      {
        for (int y = 0; y < nbvar; y++)
          cout << (*it)[y] << " ";
        cout << endl;
        // final_samples_weighted.push_back(make_pair(*it,1.0/(pivot-1)));
        total_final_samples.push_back(*it);
      }
      sample += samples_pool.size();
    }

    //////////////////////////////////////////
    // save to file
    //////////////////////////////////////////

    ofstream outfile;
    outfile.open(outpath);
    cout << "write to " << outpath << endl;
    outfile << parity_number << endl;
    for (int num = 0; num < total_final_samples.size(); num++)
    {
      for (int idx = 0; idx < total_final_samples[num].size(); idx++)
      {
        outfile << total_final_samples[num][idx];
      }
      outfile << endl;
    }
    outfile.close();

    // // compute marginals using the samples collected
    // compute_marginals(final_samples);
    // compute_marginals(total_final_samples);

    // // for(int idx = 0; idx < (int)final_samples.size(); idx++){
    // //   cout << final_samples[idx] << "\n";
    // // }

    // cout <<"Log-partition function estimate: " << (-M0*normfCPO+log10(2)*level-log10(approx_factor)*nbuckets-log10(pow(2.0,b)-1.0)*nbuckets+wcspmarkov_log)*log(10.0) << endl;
  }
  catch (IloException &ex)
  {
    cout << "Error: " << ex << endl;
  }
  env.end();
  return 0;
}

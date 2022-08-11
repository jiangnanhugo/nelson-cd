/** PAWS: PArity-based Weighted Sampler */

#include <ilcp/cpext.h>
#include <sys/time.h>

// global parity constraint for adding k XORs
#include "parity.h"
#include "parityWrapper.h"

#include "toulbar2/src/toulbar2.hpp"
#include "toulbar2/src/tb2domain.hpp"

// PAWS PARAMETERS
unsigned long samples_num = 20;
unsigned long burningIn = 0;
bool merge_singleton_factors = false;
size_t pivot = 4;
bool optimize_sol = true;

std::set < std::vector < size_t> > samples_pool;

int nbauxvars = -1;
int alpha = 1;
int b = 1;					// bits per bucket
double normfCPO;
IloInt M0;
IloInt Mmax;
double approx_factor;
	
// use ILOG's STL namespace
ILOSTLBEGIN

/**********************/

extern ostream& operator<<(ostream& os, WCSP &wcsp);

unsigned long get_seed(void) {
  struct timeval tv;
  struct timezone tzp;
  gettimeofday(&tv,&tzp);
  return (( tv.tv_sec & 0177 ) * 1000000) + tv.tv_usec;
}

void parseArgs(int argc, char **argv);
void compute_marginals(	std::vector < std::vector < size_t> > final_samples);
void compute_marginals_weighted(std::vector < std::pair <std::vector < size_t>, double> > final_samples);

//////////////////////

Cost wcsplb;
TProb wcspmarkov_log;
TProb wcspnf;

// backtrackable WCSP datastore
const int StoreSize = 16;
static Store STORE(StoreSize);

IlcIntVar Objective;
int ProblemSize = 0;
IlcIntVarArray ProblemVars;
IlcIntVarArray	AuxiliaryVars;
// mod: use long long for 64-bit systems (and make sure Makefile options match!)
//int UpperBound = (int) MAX_COST;   // best solution cost or initial global upper bound
long long UpperBound =  MAX_COST;   // best solution cost or initial global upper bound

int *BestSol = NULL;         // best solution found during the search

int nbr_solution_found_so_far = 0;
int lastConflictVar=-1;

// current IloCP instance used by libtb2.a to generate a failure
IloCP IlogSolver;

// current WeightedCSP instance used by value and variable ordering heuristics
WeightedCSP *CurrentWeightedCSP = NULL;



std::vector<int> free_ind;

// global weighted csp constraint exploiting toulbar2 propagation
class IlcWeightedCSPI : public IlcConstraintI
{
 public:
  static vector<IlcWeightedCSPI *> AllIlcWeightedCSPI;
  static int wcspCounter;
  
  IlcIntVar obj;                // objective function
  int size;                     // |vars|
  IlcIntVarArray vars;          // all Ilog variables involved in the WCSP network
  WeightedCSP *wcsp;            // WCSP network managed by ToulBar2
  Domain *unassignedVars;       // a WCSP domain containing the list of unassigned variables 
  IlcInt currentNumberOfFails;  // counter of search failures to inform ToulBar2 to reset its propagation queues and update its timestamp
  IlcRevBool synchronized;       // if IlcTrue then force a complete synchronization between Ilog and ToulBar2 variable domains and objective
 
  // unique IlcWeightedCSPI constructor
  // creates an empty WCSP and add soft constraints from a file if available
  IlcWeightedCSPI(IloCP solver,
                  IlcIntVar objective, IlcIntVarArray variables, 
                  const char *fileName = NULL);

  // destructor
  ~IlcWeightedCSPI() {
    AllIlcWeightedCSPI[wcsp->getIndex()] = NULL;
    delete wcsp;
    delete unassignedVars;
  }

  // domain synchronization between obj&vars (Ilog) and wcsp (ToulBar2)
  void synchronize();

  // if a search node failure has just occured then informs ToulBar2 to reset its propagation queues and update its timestamp
  void checkFailure() {
    if (ToulBar2::verbose >= 2) cout << "checkin failure" << endl;
    if (getSolver().getInfo(IloCP::NumberOfFails) != currentNumberOfFails) {
      if (ToulBar2::verbose >= 2) cout << "CONTRADICTION DETECTED" << endl;
      currentNumberOfFails = getSolver().getInfo(IloCP::NumberOfFails);
      wcsp->whenContradiction();
    }
  }
  
  // links the WCSP variables to the ILOG variables
  void post();

  // global propagation using WCSP propagation queues
  void propagate() {
    //cout << "TOULBAR PROP" << endl;
    checkFailure();
    if (synchronized) {
      synchronized.setValue(getSolver(), IlcFalse);
      synchronize();
    }
    if (ToulBar2::verbose >= 2) cout << "ILOG: propagate wcsp index " << wcsp->getIndex() << endl;
    if (ToulBar2::verbose >= 2)	cout << "decreasing ub in propagate"<<endl;
    //// wcsp->decreaseUb(obj.getMax()+1);
    wcsp->propagate();
  }
    
  // variable varIndex has been assigned
  void whenValue(const IlcInt varIndex) {
    checkFailure();
    if (ToulBar2::verbose >= 2) cout << "ILOG: " << vars[varIndex].getName() << " = " << vars[varIndex].getValue() << endl;
    wcsp->assign(varIndex, vars[varIndex].getValue());
    if (unassignedVars->canbe(varIndex)) {
      unassignedVars->erase(varIndex);
      if (unassignedVars->empty()) {
        assert(wcsp->verify());
        // mod: should NOT set bound on the objective here!
        //cout << "setting obj"<<endl;
        //obj.setValue(wcsp->getLb());
      }
    }
    push();   // global propagation done after local propagation
  }
    
  // check only modifications on the objective variable
  void whenRange() {
    checkFailure();
    wcsp->enforceUb(); // fail if lower bound >= upper bound and enforce NC*
    if (obj.getMax()+1 < wcsp->getUb()) {
      ////  wcsp->decreaseUb(obj.getMax()+1);
    }
    push();   // global propagation done after local propagation
  }
  
  // variable varIndex has its domain reduced
  void whenDomain(const IlcInt varIndex) {
    checkFailure();
    if (!vars[varIndex].isBound()) {
      for (IlcIntVarDeltaIterator iter(vars[varIndex]); iter.ok(); ++iter) {
        IlcInt val = *iter;
        if (ToulBar2::verbose >= 2) cout << "ILOG: " << vars[varIndex].getName() << " != " << val << endl;
        wcsp->remove(varIndex, val);
      }
      push();   // global propagation done after local propagation
    }
  }
};

vector<IlcWeightedCSPI *> IlcWeightedCSPI::AllIlcWeightedCSPI;
int IlcWeightedCSPI::wcspCounter = 0;

void tb2setvalue(int wcspId, int varIndex, Value value, void *solver)
{
  assert(wcspId < IlcWeightedCSPI::wcspCounter);
  assert(IlcWeightedCSPI::AllIlcWeightedCSPI[wcspId] != NULL);
  assert(varIndex < IlcWeightedCSPI::AllIlcWeightedCSPI[wcspId]->size);
  if (ToulBar2::verbose >= 2) cout << "TOULBAR2: x" << varIndex << " = " << value << endl;
  IlcWeightedCSPI::AllIlcWeightedCSPI[wcspId]->vars[varIndex].setValue(value);
}

void tb2removevalue(int wcspId, int varIndex, Value value, void *solver)
{
  assert(wcspId < IlcWeightedCSPI::wcspCounter);
  assert(IlcWeightedCSPI::AllIlcWeightedCSPI[wcspId] != NULL);
  assert(varIndex < IlcWeightedCSPI::AllIlcWeightedCSPI[wcspId]->size);
  if (ToulBar2::verbose >= 2) cout << "TOULBAR2: x" << varIndex << " != " << value << endl;
  IlcWeightedCSPI::AllIlcWeightedCSPI[wcspId]->vars[varIndex].removeValue(value);
}

void tb2setmin(int wcspId, int varIndex, Value value, void *solver)
{
  assert(wcspId < IlcWeightedCSPI::wcspCounter);
  assert(IlcWeightedCSPI::AllIlcWeightedCSPI[wcspId] != NULL);
  assert(varIndex < IlcWeightedCSPI::AllIlcWeightedCSPI[wcspId]->size);
  if (ToulBar2::verbose >= 2) cout << "TOULBAR2: x" << varIndex << " >= " << value << endl;
  IlcWeightedCSPI::AllIlcWeightedCSPI[wcspId]->vars[varIndex].setMin(value);
}

void tb2setmax(int wcspId, int varIndex, Value value, void *solver)
{
  assert(wcspId < IlcWeightedCSPI::wcspCounter);
  assert(IlcWeightedCSPI::AllIlcWeightedCSPI[wcspId] != NULL);
  assert(varIndex < IlcWeightedCSPI::AllIlcWeightedCSPI[wcspId]->size);
  if (ToulBar2::verbose >= 2) cout << "TOULBAR2: x" << varIndex << " <= " << value << endl;
  IlcWeightedCSPI::AllIlcWeightedCSPI[wcspId]->vars[varIndex].setMax(value);
}

//void tb2setminobj(int wcspId, int varIndex, Value value, void *solver)
void tb2setminobj(int wcspId, int varIndex, Cost value, void *solver)
{
  assert(wcspId < IlcWeightedCSPI::wcspCounter);
  assert(IlcWeightedCSPI::AllIlcWeightedCSPI[wcspId] != NULL);
  assert(varIndex == -1);
  if (ToulBar2::verbose >= 2) cout << "TOULBAR2: obj" << " >= " << value << endl;
  IlcWeightedCSPI::AllIlcWeightedCSPI[wcspId]->obj.setMin(value);
}

ILCCTDEMON1(IlcWeightedCSPWhenValueDemon, IlcWeightedCSPI, whenValue, IlcInt, varIndex);
ILCCTDEMON1(IlcWeightedCSPWhenDomainDemon, IlcWeightedCSPI, whenDomain, IlcInt, varIndex);
ILCCTDEMON0(IlcWeightedCSPWhenRangeDemon, IlcWeightedCSPI, whenRange);

void IlcWeightedCSPI::post()
{
  ToulBar2::setvalue = ::tb2setvalue;
  ToulBar2::removevalue = ::tb2removevalue;
  ToulBar2::setmin = ::tb2setmin;
  ToulBar2::setmax = ::tb2setmax;
  ToulBar2::setminobj = ::tb2setminobj;
  for (int i = 0; i < size; i++) {
    vars[i].whenValue(IlcWeightedCSPWhenValueDemon(getSolver(), this, i));
    vars[i].whenDomain(IlcWeightedCSPWhenDomainDemon(getSolver(), this, i));
  }
  obj.whenRange(IlcWeightedCSPWhenRangeDemon(getSolver(), this));
}

IlcConstraint IlcWeightedCSP(IlcIntVar objective, IlcIntVarArray variables, const char *filename)
{
  IloCP solver = objective.getSolver();
  return IlcConstraint(new (solver.getHeap()) 
                       IlcWeightedCSPI(solver, objective, variables, filename));
}
ILOCPCONSTRAINTWRAPPER3(IloWeightedCSP, solver, IloIntVar, obj, IloIntVarArray, vars,
                        const char *, filename)
{
  use(solver, obj);
  use(solver, vars);
  return IlcWeightedCSP(solver.getIntVar(obj), solver.getIntVarArray(vars), filename);
}

ILCGOAL2(IlcGuess, IlcIntVar, var, IlcInt, value)
{
  STORE.store();
  if (ToulBar2::verbose >= 2) cout << "[" << "DEPTH" << "," << Objective.getMin() << "," << UpperBound << "] Try " << var << " = " << value << endl;
  var.setValue(value);
  lastConflictVar = -1;
  return 0;
}

ILCGOAL3(IlcRefute, IlcIntVar, var, IlcInt, value, IlcInt, depth)
{
  if (ToulBar2::verbose >= 2) cout << "REFUTE" << endl;
  STORE.restore(depth);
  STORE.store();   // => store.getDepth() == getSolver().getSearchNode().getDepth()
  //if (ToulBar2::verbose >= 2) cout << "CPOFails" << getSolver().getInfo(IloCP::NumberOfFails) <<" current fails" << currentNumberOfFails << " lab" << getSolver().getInfo(IloCP::FailStatus)<< endl;
  //if (getSolver().getInfo(IloCP::NumberOfFails) == currentNumberOfFails) 
  
  ////"""""""""""
  if (optimize_sol)
    Objective.setMax(UpperBound-1);
  //else
  //	     Objective.setMax(CurrentWeightedCSP->getUb());//(Mmax);
	 
  if (ToulBar2::verbose >= 2) cout << "[" << "DEPTH" << "," << Objective.getMin() << "," << UpperBound << "] Refute " << var << " != " << value << endl;
  var.removeValue(value);
  return 0;
}

ILCGOAL0(IlcNewSolution)
{

  // if (getSolver().isInSearch() && !getSolver().isInRecomputeMode()) {
  if (getSolver().isInSearch() && !getSolver().isInReplay()) {
    //if (ToulBar2::verbose >= 0) cout << "New solution: " << Objective.getMin() << " (" << getSolver().getNumberOfFails() << " fails)" << endl;
    if (ToulBar2::verbose >= 0) cout << "New solution: " << Objective.getMin() << " log10like: " << -(Objective.getMin() + wcsplb)/wcspnf + wcspmarkov_log <<" (" << getSolver().getInfo(IloCP::NumberOfFails) << " fails)"  <<endl;
    if (ToulBar2::showSolutions) {
      cout << "Current solution: ";
      for (int i=0; i<ProblemSize; i++) {
        cout << ProblemVars[i].getValue();
        //cout << ProblemVars[i] << endl;
      }
      cout << endl;
    }
  
    //////// remove this for sampling?
    //CurrentWeightedCSP->updateUb(Objective.getMin());
    UpperBound = Objective.getMin();
  
    for (int i=0; i<ProblemSize; i++) {
      BestSol[i] = ProblemVars[i].getValue();
    }
  
    if (optimize_sol)
      {
		fail();
		return 0;
      }
	
	// else look for up to pivot solutions
	// print yvars
    /*
      cout <<"YVARS"<<endl;
      for (int i=0; i<nbauxvars ; i++) {
      cout << (Objective.getMin()  >= (IloInt) ceil(M0+(i+1)*log10(approx_factor)/normfCPO)) << " ";
      cout << AuxiliaryVars[i].getValue() <<"***";
      //AuxiliaryVars[i].setValue( (Objective.getMin()  >= (IloInt) ceil(M0+(i+1)*log10(approx_factor)/normfCPO))    );
	
      }
      cout <<endl;
	*/  
  }
  std::vector < size_t> current_sol;							
  current_sol.resize(ProblemSize+	nbauxvars );
  for (int y=0;y<ProblemSize;y++)
    current_sol[y] = ProblemVars[y].getValue();
	
  for (int y=0;y<	nbauxvars;y++)
    current_sol[ProblemSize+y] =  AuxiliaryVars[y].getValue();

  // check if solution is really different
  samples_pool.insert(current_sol);
	
  int nbr_diff_solution_foundd = samples_pool.size();
  if (nbr_diff_solution_foundd < (int)pivot)
    fail(); 
  
  //nbr_solution_found_so_far++;
  //if (nbr_solution_found_so_far<5)
  //	fail();
	
  // stop after finding a single solution
  //fail();
  return 0;
}

ILOCPGOALWRAPPER0(IloNewSolution, solver)
{
  return IlcNewSolution(solver);
}

ILCGOAL2(IlcInstantiateVar, IlcIntVar, var, IlcInt, varIndex)
{
  IlcInt value;
  int depth = STORE.getDepth();
	
  if (var.isBound())
    return 0;

  // value ordering heuristic: try the unary support first
  // value = CurrentWeightedCSP->getSupport(varIndex);
  //	value =    CurrentWeightedCSP->getBestValue(varIndex);
	
  //double r = ((double) rand() / (RAND_MAX));
  //if (r<0.5)
  //{
  IlcInt bestval = CurrentWeightedCSP->getBestValue(varIndex);
  value= (CurrentWeightedCSP->canbe(varIndex, bestval))?bestval:CurrentWeightedCSP->getSupport(varIndex);
  /*
    }
    else
    {
	value = rand() % 2;
	if (!CurrentWeightedCSP->canbe(varIndex, value))
    value = 1-value;
    }	
  */	
  if (!var.isInDomain(value))
    value = var.getMin();
  return IlcOr( IlcGuess(getSolver(), var, value),
                IlcAnd( IlcRefute(getSolver(), var, value, depth), this));
}

// variable ordering heuristic: selects the first unassigned variable with the smallest ratio current domain size divided by actual current degree in the WCSP network 
IlcInt IlcChooseMinSizeIntDivMaxDegree(const IlcIntVarArray vars)
{
  int varIndex = -1;;
  //double best = MAX_VAL - MIN_VAL;
  /*  
      for (int i=0; i<vars.getSize(); i++) if (!vars[i].isBound()) {
      // remove following "+1" when isolated variables are automatically assigned
      double heuristic = (double) CurrentWeightedCSP->getDomainSize(i) / (CurrentWeightedCSP->getDegree(i) + 1);
      if (varIndex < 0 || heuristic < best - 1./100001.) {
      best = heuristic;
      varIndex = i;
      }
      }
  */
  for (unsigned i=0; i<free_ind.size(); i++)
    if (!vars[free_ind[i]].isBound()) {
      varIndex = free_ind[i];
      break;
    }

  return varIndex;
}



IlcInt getVarMinDomainDivMaxWeightedDegreeLastConflict(const IlcIntVarArray vars)
{
  //   cout << "getVarMinDomainDivMaxWeightedDegreeLastConflict" ;
  if (lastConflictVar != -1 && !vars[lastConflictVar].isBound()) return lastConflictVar;
  int varIndex = -1;
  Cost worstUnaryCost = MIN_COST;
  double best = MAX_VAL - MIN_VAL;
  /*
    for (BTList<Value>::iterator iter = unassignedVars->begin(); iter != unassignedVars->end(); ++iter) {
    // remove following "+1" when isolated variables are automatically assigned
    double heuristic = (double) wcsp->getDomainSize(*iter) / (double) (wcsp->getWeightedDegree(*iter) + 1);
    if (varIndex < 0 || heuristic < best - 1./100001.
    || (heuristic < best + 1./100001. && wcsp->getMaxUnaryCost(*iter) > worstUnaryCost)) {
    best = heuristic;
    varIndex = *iter;
    worstUnaryCost = wcsp->getMaxUnaryCost(*iter);
    }
    }
  */
  for (int i=0; i<vars.getSize(); i++) if (!vars[i].isBound()) {
      // remove following "+1" when isolated variables are automatically assigned
      double heuristic = (double) CurrentWeightedCSP->getDomainSize(i) / (double) (CurrentWeightedCSP->getWeightedDegree(i) + 1);
      if (varIndex < 0 || heuristic < best - 1./100001.
          || (heuristic < best + 1./100001. && CurrentWeightedCSP->getMaxUnaryCost(i) > worstUnaryCost)) {
        best = heuristic;
        varIndex = i;
        worstUnaryCost =CurrentWeightedCSP->getMaxUnaryCost(i);
      }
    }
  lastConflictVar=varIndex;
  return varIndex;
}


// For Queens problem
// variable ordering heuristic: among the unassigned variables with the smallest current domain size, selects the first one with the smallest domain value
IlcChooseIndex2(IlcChooseMinSizeMin,var.getSize(),var.getMin(),IlcIntVar)

ILCGOAL1(IlcGenerateVars, IlcIntVarArray, vars)
{
  // IlcInt index = IlcChooseMinSizeIntDivMaxDegree(vars);
  //  IlcInt index = IlcChooseMinSizeMin(vars);
  IlcInt index =getVarMinDomainDivMaxWeightedDegreeLastConflict(vars);
  //  IlcInt index =RBMHeuristic(vars);
  if (index == -1) return 0;
  return IlcAnd(IlcInstantiateVar(getSolver(), vars[index], index), this);
}
ILOCPGOALWRAPPER1(IloGenerateVars, solver, IloIntVarArray, vars)
{
  return IlcGenerateVars(solver, solver.getIntVarArray(vars));
}


//////////////////////


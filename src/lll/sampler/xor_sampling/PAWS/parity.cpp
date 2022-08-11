#include "parity.h"
#include <sys/time.h>

#define DBG(x) // use this to not print debug info
//#define DBG(x) x     // use this to print debug info

#define NDBG(x) x

ILOSTLBEGIN

IloInt parity_number = 0;
IloInt parity_minlength = -1;
IloInt parity_maxlength = -1;
unsigned long parity_seed;
IloBool parity_use_given_seed = false;
IloInt parity_filterLevel = 2;
// parity_filterLevel = 0: binary representation, individual xors
// parity_filterLevel = 1: binary representation, Gaussian elimination
// parity_filterLevel = 2: CP variable representation, individual xors
IloInt parity_filterThreshold = DEFAULT_FILTER_THRESHOLD;
IloBool parity_useBitxors = true;

bool PARITY_DONT_HANDLE_RANDOM_SEED = false;

set<int> parity_associatedVars;
set<int> parity_nonBasicVars;

bool use_watched_variables = false;

static const inline int randint(const int min, const int max)
{
  // use higher order bits rather than something like min+rand()%(max-min)
  return min + (int)(((double)(max - min + 1)) * rand() / (RAND_MAX + 1.0));
}

unsigned long parity_get_seed(void)
{
  struct timeval tv;
  struct timezone tzp;
  gettimeofday(&tv, &tzp);
  return ((tv.tv_sec & 0177) * 1000000) + tv.tv_usec;
}

//-----------------------------------------------------------------------------
IlcParityConstraintI::IlcParityConstraintI(const IloCP solver, const IlcInt FilterLevel, const IlcBool UseBitxors, const IlcIntVarArray Vars, const IlcInt Number, const IlcInt MinLength, const IlcInt MaxLength, const IlcInt FilterThreshold)
    : IlcConstraintI(solver), s(solver), filterLevel(FilterLevel), useBitxors(UseBitxors), vars(Vars), number(Number), minlength(MinLength), maxlength(MaxLength), filter_threshold(FilterThreshold), inconsistent(false), largest_id_assigned_var(-1)
{

  if (number <= 0)
  {
    s.out() << "Not adding any parity constraints" << endl;
    return;
  }

  // set random seed if not already done so
  if (!PARITY_DONT_HANDLE_RANDOM_SEED)
  {
    if (!parity_use_given_seed)
      parity_seed = parity_get_seed();
    srand(parity_seed);
    PARITY_DONT_HANDLE_RANDOM_SEED = true;
  }

  // initialize parameters
  if (minlength <= 0)
  {
    if (maxlength <= 0)
    {
      // default values
      minlength = vars.getSize() / 2;
      maxlength = vars.getSize() / 2;
    }
    else
      minlength = maxlength;
  }
  else if (maxlength <= 0)
    maxlength = minlength;
  s.out() << "Adding " << number
          << " parity constraints with length between "
          << minlength << " and " << maxlength << endl;

  if (filterLevel == 2)
  {
    int max = 0;
    for (int i = 0; i < vars.getSize(); i++)
      if (vars[i].getSize() > max)
        max = vars[i].getSize();

    modulus_c = max;

    // create random XOR constraints on CP variables
    for (int i = 0; i < number; ++i)
    {
      set<int> tempIdx;
      const unsigned int length = randint(minlength, maxlength);
      while (tempIdx.size() < length)
      {
        const int var = randint(0, vars.getSize() - 1);
        tempIdx.insert(var);
      }
      IlcIntVarArray xorVars(s, length);
      int j = 0;
      for (set<int>::iterator itr = tempIdx.begin(); itr != tempIdx.end(); ++itr)
        xorVars[j++] = vars[*itr];
      const int rhs = randint(0, modulus_c - 1);
      s.add(IlcXorCPConstraint(s, xorVars, rhs, modulus_c, filter_threshold));
      NDBG(s.out() << "xorsCP[" << i << "]: ";);
      NDBG(printIndividualXor(tempIdx, rhs););
    }
    NDBG(s.out() << "Using individual CP Xors filtering" << endl;);
    return;
  }

  // create random XOR constraints on binary variables

  else if (filterLevel < 2)
  {
    // define potential binary variables as variable-value pairs
    vector<pair<int, IlcInt>> potentialBinaryVars;
    for (int i = 0; i < vars.getSize(); i++)
      ////////    for (IlcIntExpIterator it(vars[i]); it.ok(); ++it)
      ////////      potentialBinaryVars.push_back(pair<int,IlcInt> (i,*it));
      potentialBinaryVars.push_back(pair<int, IlcInt>(i, 0));

    // create random XOR constraints
    // first use all potential binary variables in xor LHS
    const int n = potentialBinaryVars.size();
    vector<set<int>> tempIdx(number);
    vector<bool> tempIsEven(number);
    set<int> activeBinaryVars;
    for (int i = 0; i < number; ++i)
    {
      const unsigned int length = randint(minlength, maxlength);
      while (tempIdx[i].size() < length)
      {
        //////////////////////////////////////////////////////////////
        const int var = randint(0, n - 1);
        ////////////////////////////////////////////////////////
        ///  const int var = randint(0, vars.getSize()-1);
        tempIdx[i].insert(var);
        activeBinaryVars.insert(var);
      }
      tempIsEven[i] = (randint(0, 1) == 0);
    }

    // now shrink to active binary variables and rename to [0 .. n_active-1]
    const int n_active = activeBinaryVars.size();
    //  s.out() << "Number of potential binaryVars = " << n << endl;
    //  s.out() << "Number of binaryVars created   = " << n_active << " [0.." << n_active - 1 << ']' << endl;
    binaryVars = IlcIntVarArray(s, n_active, 0, 1);
    int i = 0;
    for (set<int>::iterator itr = activeBinaryVars.begin(); itr != activeBinaryVars.end(); ++itr)
    {
      const int var = potentialBinaryVars[*itr].first;
      // const int val = potentialBinaryVars[*itr].second;
      //    s.add( binaryVars[i] == (vars[var] == val) );
      binaryVars[i] = vars[var];
      reverseVarMap.insert(pair<int, int>(*itr, i));
      ++i;
    }

    //////////////////////////
    // if binary individual xors are desired (filterLevel == 0), add them to the model here and stop
    if (filterLevel == 0)
    {
      for (int i = 0; i < number; ++i)
      {
        IlcIntVarArray xorVars(s, tempIdx[i].size());
        int j = 0;
        for (set<int>::iterator itr = tempIdx[i].begin(); itr != tempIdx[i].end(); ++itr)
          xorVars[j++] = binaryVars[reverseVarMap[*itr]];
        s.add(IlcXorConstraint(s, xorVars, tempIsEven[i]));
        NDBG(s.out() << "xors[" << i << "]: ";);
        NDBG(printIndividualXor(tempIdx[i], tempIsEven[i] ? 0 : 1););
      }
      NDBG(s.out() << "Using individual XOR filtering" << endl;);
      return;
    }

    //////////////////////////
    // individual xors NOT wanted; continue toward Gaussian elimination
    xors.resize(number);
    if (number > MAXBITSETSIZE && useBitxors)
    {
      cerr << "ERROR: number of XOR constraints, " << number << ", exceeds the max bitset size " << MAXBITSETSIZE << endl;
      exit(1);
    }
    for (int i = 0; i < number; ++i)
    {
      xors[i] = (useBitxors ? (XorCons *)new XorConsBitset() : (XorCons *)new XorConsSet);
      for (set<int>::iterator itr = tempIdx[i].begin(); itr != tempIdx[i].end(); ++itr)
        ////////////////////////////////////////////////
        xors[i]->addIdx(reverseVarMap[*itr]);
      // xors[i].idx.insert(*itr);
      xors[i]->parity = (tempIsEven[i] ? even : odd);
    }
    DBG(printAllXors());
    if (useBitxors)
      cout << "Using Gaussian Elimination with XORs as bitsets for filtering parity constraints" << endl;
    else
      cout << "Using Gaussian Elimination with XORs as set<int> for filtering parity constraints" << endl;

    // diagonalize the constraints (triangularize both ways)
    // using Gaussian elimination
    for (int i = 0; i < number; ++i)
    {
      if (xors[i]->isEmpty())
      {
        if (xors[i]->parity == odd)
        {
          NDBG(cout << "Inconsistency detected!" << endl;);
          // binaryVars[0].removeValue(0);
          // binaryVars[0].removeValue(1);
          inconsistent = true;
          // s.fail() ???   (probably not in the constructor)
          return;
        }
        else
        {
          // linearly dependent xor constraint; remove it
          xors.erase(xors.begin() + i);
          --i;
          --number;
          continue;
        }
      }
      xors[i]->associatedVar = xors[i]->getFirstIdx();
      // cout << "ass var[" <<i <<"]=" <<  xors[i].associatedVar <<  endl;
      for (int j = 0; j < number; ++j)
      {
        if (j == i)
          continue;
        if (xors[j]->isPresent(xors[i]->associatedVar))
          *(xors[j]) += *(xors[i]);
      }
    }
    DBG(cout << "After diagonalization:" << endl;);
    DBG(printAllXors());

    // isActiveXor = IlcRevBoolArray(s, number);
    isActiveXor.resize(number);
    for (int i = 0; i < number; i++)
      isActiveXor[i] = new IlcRevBool(s, IlcTrue);

    // store the set of associated variables in a global set<int> for external access
    parity_associatedVars.clear();
    for (int i = 0; i < number; i++)
    {
      // cout << "xor " <<  i<<" with watched basic var[" <<xors[i]->associatedVar<<"]"<< endl;
      parity_associatedVars.insert(xors[i]->associatedVar);
    }
    parity_nonBasicVars.clear();
    // set non-basic variables
    for (int i = 0; i < number; i++)
      for (int j = vars.getSize() - 1; j >= 0; j--)
      {
        if (xors[i]->isPresent(j) && parity_associatedVars.count(j) == 0)
        {
          xors[i]->NonBasicVar = j;
          parity_nonBasicVars.insert(j);
          // cout << "xor " <<  i<<" with watched non basic var[" <<j<<"]"<< endl;
          break;
        }
      }

    // compute the largest non-associated variable
    for (int i = 0; i < vars.getSize(); i++)
    {
      if ((i > largest_id_assigned_var) && parity_associatedVars.find(i) == parity_associatedVars.end())
        largest_id_assigned_var = i;
    }
    // cout << "largest var to be assigned bf triggering GE:" << largest_id_assigned_var << endl;
  }
}

//-----------------------------------------------------------------------------
IlcParityConstraintI::~IlcParityConstraintI()
{
  // delete created memory
  for (unsigned i = 0; i < xors.size(); i++)
    delete xors[i];
}

//-----------------------------------------------------------------------------

ILCCTDEMON0(IlcParityPostDemon, IlcParityConstraintI, postCst);

void IlcParityConstraintI::postCst()
{
  this->push(1);
}

void IlcParityConstraintI::post()
{

  if (use_watched_variables)
  {
    if ((filterLevel == 1) && number > 0)
      for (int i = 0; i < binaryVars.getSize(); i++)
        if (parity_associatedVars.count(i) > 0 || parity_nonBasicVars.count(i) > 0)
        {
          // cout << "demon on  " <<  i<< "("<<parity_associatedVars.count(i) << ","<< parity_nonBasicVars.count(i) <<")"<<endl;
          binaryVars[i].whenDomain(IlcParityPostDemon(getSolver(), this));
        }
  }
  else
  {

    if ((filterLevel == 1) && number > 0)
      for (int i = 0; i < binaryVars.getSize(); i++)
        binaryVars[i].whenDomain(IlcParityPostDemon(getSolver(), this));
  }
}

//-----------------------------------------------------------------------------
void IlcParityConstraintI::propagate()
{
  //  DBG(cout << "Propagating ParityConstraint because of binaryVars["
  //       << this->binaryVars.getIndexValue() << ']' <<endl;);
  if ((filterLevel != 1) || number <= 0)
    return;
  /*
  for (int i =0;i<vars.getSize(); i++)
    cout << " var  " << i << "size" <<vars[i].getSize() << endl;
  */
  if (inconsistent)
    s.fail();

  // cout << "largest var" << largest_id_assigned_var << "," <<vars[largest_id_assigned_var].isBound() << endl;

  //   if (!vars[largest_id_assigned_var].isBound())
  //  return;

  // cout << "PARITY PROP" << this->binaryVars.getIndexValue() << endl;
  DBG(printAllXors());

  // perform Gaussian elimination
  DBG(cout << "perform Gaussian elimination" << endl;);
  for (int i = 0; i < number; ++i)
  {
    if (isActiveXor[i]->getValue() == IlcFalse)
      continue;

    if (use_watched_variables && (xors[i]->associatedVar != -1) && (xors[i]->NonBasicVar != -1) && !binaryVars[xors[i]->associatedVar].isBound() && !binaryVars[xors[i]->NonBasicVar].isBound())
    {
      continue;
    }

    const evaluate_t result = evaluateXor(i);
    if (result.numActive == 0)
    {
      // no active vars; degenerate xor
      xors[i]->associatedVar = -1;
      if (result.valueLHS != xors[i]->parity)
      {
        s.fail();
        return;
      }
      // don't re-evaluate xor[i] in this subtree
      isActiveXor[i]->setValue(s, IlcFalse);
      continue;
    }

    if (result.numActive == 1)
    {
      DBG(cout << "   pruning domain value "
               << (result.valueLHS == xors[i]->parity ? 1 : 0)
               << " for var " << result.firstActive << endl;);
      binaryVars[result.firstActive].removeValue(result.valueLHS == xors[i]->parity ? 1 : 0);
    }

    // **** stefano mod
    // check if we need to choose a new nonbasic var
    if (use_watched_variables)
    {
      if ((xors[i]->NonBasicVar != -1) && binaryVars[xors[i]->NonBasicVar].isBound())
      {
        if (xors[i]->isPresent(result.firstNonBasis) && parity_associatedVars.count(result.firstNonBasis) == 0 && !binaryVars[result.firstNonBasis].isBound())
        {
          xors[i]->NonBasicVar = result.firstNonBasis;
          cout << "nonbasic switching in " << result.firstNonBasis << endl;
          binaryVars[result.firstNonBasis].whenDomain(IlcParityPostDemon(getSolver(), this));
        }
        else
        {
          // cout << "+" ;
          xors[i]->NonBasicVar = -1;

          /*
          //inefficient, just check if it works..
          for (int j=vars.getSize()-1; j>=0; j--) {
          if (xors[i]->isPresent(j) && parity_associatedVars.count(j)==0 && !binaryVars[j].isBound())
            {
            xors[i]->NonBasicVar = j;
            //cout << "xor " <<  i<<" with watched non basic var[" <<j<<"]"<< endl;
            break;
            }
            }
            cout << "valnonbasic = " <<xors[i]->NonBasicVar <<" ";
            */
        }
      }
    }
    // **** end mod

    if ((xors[i]->associatedVar != -1) && !binaryVars[xors[i]->associatedVar].isBound())
      continue;

    // **** stefano mod
    // keep track of current set of associated=basic variables
    if (use_watched_variables)
    {
      parity_associatedVars.erase(xors[i]->associatedVar);
      parity_associatedVars.insert(result.firstActive);
      cout << "switching in " << result.firstActive << endl;
      binaryVars[result.firstActive].whenDomain(IlcParityPostDemon(getSolver(), this));
    }
    // **** end mod

    xors[i]->associatedVar = result.firstActive;

    // **** stefano mod
    // check if we need to choose a new nonbasic var
    if (use_watched_variables && xors[i]->associatedVar == xors[i]->NonBasicVar)
    {
      if (xors[i]->isPresent(result.firstNonBasis) && parity_associatedVars.count(result.firstNonBasis) == 0 && !binaryVars[result.firstNonBasis].isBound())
      {
        xors[i]->NonBasicVar = result.firstNonBasis;
      }
      else
      {
        // cout << "-" ;
        xors[i]->NonBasicVar = -1;
        // inefficient, just check if it works..
        for (int j = vars.getSize() - 1; j >= 0; j--)
        {
          if (xors[i]->isPresent(j) && parity_associatedVars.count(j) == 0 && !binaryVars[j].isBound())
          {
            xors[i]->NonBasicVar = j;
            binaryVars[j].whenDomain(IlcParityPostDemon(getSolver(), this));
            cout << "non-basic demon on var " << j << endl;
            // cout << "xor " <<  i<<" with watched non basic var[" <<j<<"]"<< endl;
            break;
          }
        }
        // cout << "valnonbasic = " <<xors[i]->NonBasicVar <<" ";
      }
    }
    // **** end mod

    // eliminate associated_var from all remaining xors
    for (int j = 0; j < number; ++j)
    {
      if (j == i)
        continue;
      if (xors[j]->isPresent(xors[i]->associatedVar))
      {
        (*xors[j]) += (*xors[i]);
        // **** stefano mod
        // keep track of current set of associated=basic variables
        if (use_watched_variables)
        {

          if ((xors[j]->NonBasicVar == -1) || (!xors[j]->isPresent(xors[j]->NonBasicVar)))
          {
            // cout << "*" ;
            xors[j]->NonBasicVar = -1;
            // inefficient, just check if it works..
            for (int k = vars.getSize() - 1; k >= 0; k--)
            {
              if (xors[j]->isPresent(k) && parity_associatedVars.count(k) == 0 && !binaryVars[k].isBound())
              {
                xors[j]->NonBasicVar = k;
                binaryVars[k].whenDomain(IlcParityPostDemon(getSolver(), this));
                // cout << "xor " <<  i<<" with watched non basic var[" <<j<<"]"<< endl;
                break;
              }
            }
          }
        }
        // **** end mod
      }
    }
  }

  /*
   // do another round of filtering
   for (int i=0; i<number; ++i) {
   // **** stefano mod
   // keep track of current set of associated=basic variables
      if (use_watched_variables && (xors[i]->associatedVar != -1)  && (xors[i]->NonBasicVar != -1) && !binaryVars[xors[i]->associatedVar].isBound() && !binaryVars[xors[i]->NonBasicVar].isBound() )
     {
     continue;
     }
   // **** end mod

     const evaluate_t result = evaluateXor(i);
     if (result.numActive == 1) {
       DBG(cout << "   pruning domain value "
           << (result.valueLHS==xors[i]->parity ? 1 : 0)
           << " for var " << xors[i]->associatedVar << endl;);
       binaryVars[xors[i]->associatedVar].removeValue(result.valueLHS==xors[i]->parity ? 1 : 0);
     }
   }
  */

  DBG(cout << "Result:" << endl;);
  DBG(printAllXors(););
}

//-----------------------------------------------------------------------------
IlcParityConstraintI::evaluate_t IlcParityConstraintI::evaluateXor(const int xor_num) const
{
  evaluate_t result;
  if (useBitxors)
  {
    const bitsetN &idxbits = xors[xor_num]->idxbits;
    for (unsigned i = 0; i < idxbits.size(); i++)
    {
      if (idxbits.test(i) == false)
        continue;
      if (!binaryVars[i].isInDomain(0))
        result.valueLHS ^= odd;
      else if (binaryVars[i].isInDomain(1))
      {
        ++result.numActive;
        if (result.numActive == 1)
          result.firstActive = i;
        else if (result.numActive == 2)
        {
          result.firstNonBasis = i;
          break;
        }
      }
    }
  }
  else
  {
    const set<int> &idx = xors[xor_num]->idx;
    for (set<int>::iterator itr = idx.begin(); itr != idx.end(); ++itr)
    {
      if (!binaryVars[*itr].isInDomain(0))
        result.valueLHS ^= odd;
      else if (binaryVars[*itr].isInDomain(1))
      {
        ++result.numActive;
        if (result.numActive == 1)
          result.firstActive = *itr;
        else if (result.numActive == 2)
        {
          result.firstNonBasis = *itr;
          break;
        }
      }
    }
  }
  return result;
}

//-----------------------------------------------------------------------------
void IlcParityConstraintI::printXor(const int xor_num, ostream &os) const
{
  os << "xors[" << xor_num << "]: ";
  const XorCons &cons = *(xors[xor_num]);
  if (cons.isEmpty())
  {
    if (cons.parity == even)
      os << "Empty satisfied constraint (RHS == even)" << endl;
    else
      os << "Empty violated constraint (RHS == odd)" << endl;
    return;
  }
  if (useBitxors)
  {
    bool isFirstVar = true;
    for (int var = 0; var < (int)cons.idxbits.size(); var++)
    {
      if (!cons.idxbits.test(var))
        continue;
      if (!isFirstVar)
        os << " + ";
      isFirstVar = false;
      if (var == cons.associatedVar)
        os << '*' << var << '*';
      else
        os << var;
      if (!binaryVars[var].isInDomain(0))
        os << "(1)";
      if (!binaryVars[var].isInDomain(1))
        os << "(0)";
    }
  }
  else
  {
    const set<int> &idx = cons.idx;
    set<int>::iterator itr = idx.begin();
    const int firstVar = *itr;
    for (; itr != idx.end(); ++itr)
    {
      const int var = *itr;
      if (var != firstVar)
        os << " + ";
      if (var == cons.associatedVar)
        os << '*' << var << '*';
      else
        os << var;
      if (!binaryVars[var].isInDomain(0))
        os << "(1)";
      if (!binaryVars[var].isInDomain(1))
        os << "(0)";
    }
  }
  os << " = " << cons.parity << endl;
}

//-----------------------------------------------------------------------------
void IlcParityConstraintI::printAllXors(ostream &os) const
{
  for (int i = 0; i < number; ++i)
    printXor(i, os);
}

//-----------------------------------------------------------------------------
void IlcParityConstraintI::printIndividualXor(const set<int> &idx, const int rhs, ostream &os)
{
  set<int>::iterator itr = idx.begin();
  if (itr == idx.end())
  {
    if (rhs == 0)
      os << "Empty satisfied constraint" << endl;
    else
      os << "Empty violated constraint" << endl;
    return;
  }

  if (filterLevel < 2)
  {
    const int firstVar = reverseVarMap[*itr];
    for (; itr != idx.end(); ++itr)
    {
      const int var = reverseVarMap[*itr];
      if (var != firstVar)
        os << " + ";
      os << var;
      if (!binaryVars[var].isInDomain(0))
        os << "(1)";
      if (!binaryVars[var].isInDomain(1))
        os << "(0)";
    }
    os << " = " << (rhs == 0 ? even : odd) << endl;
  }
  else
  {
    const int firstVar = *itr;
    for (; itr != idx.end(); ++itr)
    {
      const int var = *itr;
      if (var != firstVar)
        os << " + ";
      os << var;
      if (vars[var].isBound())
        os << vars[var];
    }
    os << " = " << rhs << endl;
  }
}

////////////////////////////////

void parseParityArgs(int &argc, char **argv)
{
  // this method eats up all arguments that are relevant for the
  // parity constraint, and returns the rest in argc, argv

  char residualArgv[argc][64];
  strcpy(residualArgv[0], argv[0]);
  int residualArgc = 1;

  for (int argIndex = 1; argIndex < argc; ++argIndex)
  {
    if (!strcmp(argv[argIndex], "-paritylevel"))
    {
      argIndex++;
      parity_filterLevel = atol(argv[argIndex]);
    }
    else if (!strcmp(argv[argIndex], "-paritythreshold"))
    {
      argIndex++;
      parity_filterThreshold = atol(argv[argIndex]);
      if (parity_filterThreshold < 2)
      {
        cerr << "ERROR: paritythreshold must be at least 2." << endl;
        exit(1);
      }
    }
    else if (!strcmp(argv[argIndex], "-nobitsets"))
    {
      parity_useBitxors = false;
    }
    else if (!strcmp(argv[argIndex], "-number"))
    {
      argIndex++;
      parity_number = (IloInt)atol(argv[argIndex]);
    }
    else if (!strcmp(argv[argIndex], "-minlength"))
    {
      argIndex++;
      parity_minlength = (IloInt)atol(argv[argIndex]);
    }
    else if (!strcmp(argv[argIndex], "-maxlength"))
    {
      argIndex++;
      parity_maxlength = (IloInt)atol(argv[argIndex]);
    }
    else if (!strcmp(argv[argIndex], "-seed") && !PARITY_DONT_HANDLE_RANDOM_SEED)
    {
      argIndex++;
      parity_seed = atol(argv[argIndex]);
      parity_use_given_seed = true;
    }
    else
    {
      // save this option to be returned back
      strcpy(residualArgv[residualArgc++], argv[argIndex]);
    }
  }

  argc = residualArgc;
  for (int i = 1; i < argc; ++i)
  {
    // free(argv[i]);
    argv[i] = new char[strlen(residualArgv[i]) + 1];
    strcpy(argv[i], residualArgv[i]);
  }
}

void printParityUsage(ostream &os = cout)
{
  os << "ParityConstraint options:" << endl
     << "   -paritylevel        0: binary individual Xors," << endl
     << "                       1: binary Gaussian elimination, " << endl
     << "                       2: non-binary individual Xors (default)" << endl
     << "   -paritythreshold    >= 2, for individual Xors (default: 3)" << endl
     << "   -nobitsets          do not use bitsets with Gauss Elim (default: use bitsets)," << endl
     << "   -number             Number of random XORs (default: 0)" << endl
     << "   -minlength          Minlength of XORs (default: nvars/2)" << endl
     << "   -maxlength          Maxlength of XORs (default: nvars/20)" << endl;
  if (!PARITY_DONT_HANDLE_RANDOM_SEED)
    os << "   -seed               Random seed" << endl;
  else
  {
    os << "   -seed               Disabled; to enable, remove from the model" << endl
       << "                         PARITY_DONT_HANDLE_RANDOM_SEED = true" << endl;
  }
  os << endl;
}

#include "xors_cp.h"
#include <vector>

#define DBG(x)       // use this to not print debug info
//#define DBG(x) x     // use this to print debug info

#define NDBG(x) x


//-----------------------------------------------------------------------------
IlcXorCPConstraintI::IlcXorCPConstraintI(const IloCP solver, const IlcIntVarArray Vars, const IlcInt Rhs, const IlcInt Mod_c, const IlcInt FilterThreshold)
  : IlcConstraintI(solver), s(solver), vars(Vars), rhs(Rhs), modulus_c(Mod_c), num_watched(FilterThreshold <= Vars.getSize() ? FilterThreshold : Vars.getSize()) {

  watched = new int[num_watched];
  for (int i=0; i<num_watched; ++i)
    watched[i] = i;
}


//-----------------------------------------------------------------------------
IlcXorCPConstraintI::~IlcXorCPConstraintI(){
  // delete created memory 
  delete [] watched;
}

//-----------------------------------------------------------------------------

ILCCTDEMON0(IlcXorCPPostDemon, IlcXorCPConstraintI, postCst);

void IlcXorCPConstraintI::postCst() {
  this->push(1);
}

void IlcXorCPConstraintI::post() {
  for (int i=0; i<vars.getSize(); i++)
    vars[i].whenDomain( IlcXorCPPostDemon(getSolver(), this) );
}

//-----------------------------------------------------------------------------
void IlcXorCPConstraintI::propagate() {
  if (watched[num_watched-1] != -1) {    // i.e. no duplicate watched vars
    bool nothing_to_do = true;
    for (int j=0; j<num_watched; ++j) {
      if (vars[watched[j]].isBound()) {
        nothing_to_do = false;
        break;
      }
    }
    if (nothing_to_do) {   // all watched vars are free
      return;
    }
  }

  //DBG(s.out() << "Propagating XorCPConstraint because of vars["
  //    << vars.getIndexValue() << ']' <<endl;);
  DBG(
  int j = 0;
  for (; j<vars.getSize(); j++) {
    if (vars[j].isInProcess()) {
      s.out() << "Propagating XorCPConstraint because of vars[" << j << ']' <<endl;
      break;
    }
  }
  if (j == vars.getSize())
    s.out() << "Propagating XorCPConstraint because of vars[unknown]" <<endl;
  print();
  );
  
  evaluate_t result = evaluate();

  if (result.numActive == 0) {
    if (result.valueLHS != rhs)
	{
 // s.out() << "failing" <<endl;
      //s.fail();
 s.fail();
	}
  }
  else if (result.numActive == 1) {
    const int value = (rhs - result.valueLHS + modulus_c) % modulus_c;
    int var_idx = watched[0];
    DBG(s.out() << "   fixing variable  " << var_idx
        << " to " << value << endl;);
    vars[var_idx].setValue(value);
  }
  else if (result.numActive >= 2 && result.numActive < num_watched) {    
    DBG(cout << "Try to filter variables " << endl;
        for(int i=0; i<result.numActive; i++)
        cout << " vars[" << watched[i] << "] = " << vars[watched[i]] << endl;
        print(););


    const int value = (rhs - result.valueLHS + modulus_c) % modulus_c;
    if (result.numActive < WHEN_GRAPH_FILTERING_IS_BETTER)
      filterBruteForce(result.numActive, value);
    else
      filterGraphBased(result.numActive, value);


    DBG(cout << "After filtering: " << endl;
        for(int i=0; i<result.numActive; i++)
        cout << " vars[" << watched[i] << "] = " << vars[watched[i]] << endl;
        print(););
  }
}

//-----------------------------------------------------------------------------
IlcXorCPConstraintI::evaluate_t IlcXorCPConstraintI::evaluate(void) {
  const int length = vars.getSize();
  const int startIdx = 0;
  
  evaluate_t result;
  int i = startIdx;
  do {
    if (vars[i].isBound())
      result.valueLHS += vars[i].getValue();
    else {
      watched[result.numActive++] = i;
      if (result.numActive == num_watched)
        break;
    }
    i = (i+1) % length;
  } while (i != startIdx);
  result.valueLHS %= modulus_c;

  // clear up the remaining watched array
  for (int j=result.numActive; j<num_watched; ++j)
    watched[j] = -1;

  return result;
}

//-----------------------------------------------------------------------------
void IlcXorCPConstraintI::filterGraphBased(const int num_active, const int target_value) {
  // filter: vars[watched[0]] + ... + vars[watched[num_active-1]] = target_value

  vector<int> in_edges[num_active+1][modulus_c];
  int out_degree[num_active][modulus_c];

  memset(out_degree, 0, num_active*modulus_c*sizeof(int));

  // create the initial graph
  for (int i=0; i<num_active; ++i) {
    for (int partial_sum=0; partial_sum < modulus_c; ++partial_sum) {
      if ((i==0 && partial_sum==0) || (i>0 && !in_edges[i][partial_sum].empty())) {
        for (IlcIntExpIterator it(vars[watched[i]]); it.ok(); ++it) {
          const int next_partial_sum = (partial_sum + *it) % modulus_c;
          ++out_degree[i][partial_sum];
          in_edges[i+1][next_partial_sum].push_back(partial_sum);
        }
      }
    }
  }

  // delete edges that do not lead to target_value
  for (int i=num_active; i>=0; --i) {
    for (int partial_sum=0; partial_sum < modulus_c; ++partial_sum) {
      if ((i==num_active && partial_sum!=target_value) || (i<num_active && out_degree[i][partial_sum]==0)) {
        vector<int> & predecessors = in_edges[i][partial_sum];
        for (unsigned int k=0; k<predecessors.size(); ++k)
          --out_degree[i-1][predecessors[k]];
        predecessors.clear();
      }
    }
  }

  // filter domains
  for (int i=0; i<num_active; ++i) {
    IlcIntSet support_values(s, 0, modulus_c - 1, IlcFalse);
    for (int next_partial_sum=0; next_partial_sum < modulus_c; ++next_partial_sum) {
      vector<int> & predecessors = in_edges[i+1][next_partial_sum];
      for (unsigned int k=0; k<predecessors.size(); ++k)
        support_values.add((next_partial_sum - predecessors[k] + modulus_c) % modulus_c);
    }
    vars[watched[i]].setDomain( support_values );
  }
}

//-----------------------------------------------------------------------------
void IlcXorCPConstraintI::filterBruteForce(const int num_active, const int target_value) {
  // filter: vars[watched[0]] + ... + vars[watched[num_active-1]] = target_value

  // move watched var with max domain size to the end
  int max_domain_size = vars[watched[num_active-1]].getSize();
  int max_domain_var_idx = num_active-1;
  bool do_swap = false;
  for (int j=num_active-2; j>=0; --j) {
    if (vars[watched[j]].getSize() > max_domain_size) {
      max_domain_size = vars[watched[j]].getSize();
      max_domain_var_idx = j;
      do_swap = true;
    }
  }
  if (do_swap) {
    int tmp_idx = watched[max_domain_var_idx];
    watched[max_domain_var_idx] = watched[num_active-1];
    watched[num_active-1] = tmp_idx;
  }

  // compute support
  bool support[num_active][modulus_c];
  memset(support, 0, num_active*modulus_c*sizeof(bool));
  IlcIntArray values(s, num_active-1);
  IlcInt value_sum = 0;
  for (int i=0; i<num_active-1; ++i) {
    values[i] = vars[watched[i]].getMin();
    value_sum += values[i];
  }
  do {
    const IlcInt residual_value = (target_value - value_sum + (num_active-1)*modulus_c) % modulus_c;
    if (vars[watched[num_active-1]].isInDomain(residual_value)) {
      for (int j=0; j<num_active-1; ++j)
        support[j][values[j]] = true;
      support[num_active-1][residual_value] = true;
    }
  } while (get_next_values(values, value_sum) == true);

  // filter
  for (int j=0; j<num_active; ++j) {
    IlcIntSet support_values(s, 0, modulus_c - 1, IlcFalse);
    for (int k=0; k<modulus_c; ++k)
      if (support[j][k])
        support_values.add(k);
    vars[watched[j]].setDomain( support_values );
  }
}

//-----------------------------------------------------------------------------
bool IlcXorCPConstraintI::get_next_values(IlcIntArray values, IlcInt & value_sum) const {
  // computes the next set of values for values[]
  // also modifies value_sum to hold the sum of the new values
  const int length = values.getSize();

  for (int idx=length-1; idx>=0; --idx) {
    const int var_idx = watched[idx];
    IlcInt next_value = vars[var_idx].getNextHigher(values[idx]);
    if (next_value != values[idx]) {
      value_sum = value_sum - values[idx] + next_value;
      values[idx] = next_value;
      break;
    }
    else {
      // already at the max value; wrap around
      next_value = vars[var_idx].getMin();
      value_sum = value_sum - values[idx] + next_value + modulus_c;
      values[idx] = next_value;
      value_sum %= modulus_c;
      if (idx == 0)
        return false;
    }
  }
  value_sum %= modulus_c;
  return true;
}

//-----------------------------------------------------------------------------
void IlcXorCPConstraintI::print(ostream & os) const {
  if (vars.getSize() == 0) {
    if (rhs==0)
      os << "Empty satisfied constraint" << endl;
    else
      os << "Empty violated constraint" << endl;
    return;
  }
  for (int i=0; i<vars.getSize(); ++i) {
    if (i != 0)
      os << " + ";
    bool is_watched = false;
    for (int j=0; j<num_watched; ++j)
      if (watched[j] == i)
        is_watched = true;
    if (is_watched)
        os << '*' << i << '*';
    else
      os << i;
    if (vars[i].isBound())
      os << vars[i];
  }
  os << " = " << rhs << endl;
}

//-----------------------------------------------------------------------------
ostream & operator << (ostream & os, const IlcXorCPConstraintI cons) {
  cons.print();
  return os;
}

//-----------------------------------------------------------------------------
IlcConstraint IlcXorCPConstraint(IloCP solver, IlcIntVarArray vars, IlcInt rhs, IlcInt modulus_c, IlcInt filter_threshold) {
  
  return new (solver.getHeap()) IlcXorCPConstraintI(solver, vars, rhs, modulus_c, filter_threshold);
}




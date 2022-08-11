#ifndef PARITY
#define PARITY

#include <ilcp/cpext.h>
#include <vector>
#include <set>
#include <bitset>
#include <algorithm>
#ifndef _MSC_VER
  #include <tr1/unordered_map>
#else
  #include <unordered_map>
#endif

#include "xors.h"
#include "xors_cp.h"

using namespace std;
using std::tr1::unordered_map;



extern IloInt parity_number;
extern IloInt parity_minlength;
extern IloInt parity_maxlength;
extern unsigned long parity_seed;
extern IloInt parity_filterLevel;
extern IloInt parity_filterThreshold;
extern IloBool parity_useBitxors;

extern bool PARITY_DONT_HANDLE_RANDOM_SEED;

extern set<int> parity_associatedVars;

#define MAXBITSETSIZE 1024
typedef bitset<MAXBITSETSIZE> bitsetN;
inline bitsetN & operator += (bitsetN & A, const bitsetN & B) {
  A += B;
  return A;
}

void printParityUsage(ostream & os);
void parseParityArgs(int & argc, char **argv);



/////////////////////////////////////////

class XorCons {
 public:
  set<int> idx;
  bitsetN  idxbits;
  parity_t parity;
  int associatedVar;
  int NonBasicVar;
  
  XorCons() : parity(even), associatedVar(-1), NonBasicVar(-1) {}

  virtual bool       isEmpty() const { assert(false); return true; }
  virtual bool       isPresent(const int i) const { assert(false); return false; }
  virtual void       addIdx(const int i) { assert(false); }
  virtual int        getFirstIdx() const { assert(false); return -1; }
  virtual XorCons &  operator += (const XorCons & cons2) { assert(false); return *this; }
};

class XorConsSet : public XorCons {
 public:
  bool       isEmpty() const { return idx.empty(); }
  bool       isPresent(const int i) const { return idx.find(i) != idx.end(); }
  void       addIdx(const int i) { idx.insert(i); }
  int        getFirstIdx() const { assert(!idx.empty()); return *(idx.begin()); }
  XorConsSet & operator += (const XorCons & cons2);
};

class XorConsBitset : public XorCons {
 public:
  bool       isEmpty() const { return idxbits.none(); }
  bool       isPresent(const int i) const { return idxbits.test(i); }
  void       addIdx(const int i) { idxbits.set(i); }
  int        getFirstIdx() const { 
    unsigned i=0;
    while (i<idxbits.size() && !idxbits.test(i)) i++;
    assert(i!=idxbits.size()); return i;
  }
  XorConsBitset & operator += (const XorCons & cons2) {
    idxbits ^= cons2.idxbits;
    parity ^= cons2.parity;
    return *this;
  }
};



//*****************************************************************************
// The parity constraint
//*****************************************************************************

class IlcParityConstraintI: public IlcConstraintI {
 private:
  const IloCP s;
  const int filterLevel;
  const bool useBitxors;
	
  const IlcIntVarArray vars;
  IlcIntVarArray binaryVars; // parity constraints will be defined on these variables

  IlcInt number;
  IlcInt minlength;
  IlcInt maxlength;
  vector<XorCons*> xors;
  vector<IlcRevBoolPtr> isActiveXor;
  int modulus_c;
  int filter_threshold;

  bool inconsistent;
  IloInt largest_id_assigned_var; 

  unordered_map<int,int> reverseVarMap;

  struct evaluate_t;
  evaluate_t evaluateXor(const int xor_num) const;

  void printXor(const int xor_num, ostream & os = cout) const;
  void printAllXors(ostream & os = cout) const;
  void printIndividualXor(const set<int> & idx, const int rhs, ostream & os = cout);

 public:
  IlcParityConstraintI(const IloCP solver, const IlcInt FilterLevel, const IlcBool useBitxors, const IlcIntVarArray Vars, const IlcInt Number, const IlcInt MinLength, const IlcInt Maxlength, const IlcInt FilterThreshold);

  ~IlcParityConstraintI();

  virtual void post();
  virtual void propagate();

  IlcRevBool _binaryVarsIsModified;
  void memorizeThatBinaryVarsIsModified();
  void postCst();
};

IlcConstraint IlcParityConstraint(IloCP solver, IlcInt filterLevel, IlcBool useBitxors, IlcIntVarArray vars, IlcInt number, IlcInt minlength, IlcInt maxlength, IlcInt filterThreshold = DEFAULT_FILTER_THRESHOLD);


IlcConstraint IlcParityConstraint(IloCP solver, IlcIntVarArray vars);


struct IlcParityConstraintI::evaluate_t {
  int firstActive;
  int firstNonBasis;
  int numActive;
  vector<int> activeIdx;
  parity_t valueLHS;
  evaluate_t() : firstActive(-1), numActive(0), valueLHS(even) {}
};



///////////////////////////////////////////
////   INLINE METHODS  ////////////////////
///////////////////////////////////////////

inline XorConsSet & XorConsSet::operator += (const XorCons & cons2) {
  set<int> idxcpy(idx);
  idx.clear();
  set_symmetric_difference(idxcpy.begin(), idxcpy.end(), cons2.idx.begin(), cons2.idx.end(), inserter(idx, idx.begin()));
  parity ^= cons2.parity;
  return *this;
}

inline IlcConstraint IlcParityConstraint(IloCP solver, IlcInt filterLevel, IlcBool useBitxors, IlcIntVarArray vars, IlcInt number, IlcInt minlength, IlcInt maxlength, IlcInt filterThreshold) {
  return new (solver.getHeap()) IlcParityConstraintI(solver, filterLevel, useBitxors, vars, number, minlength, maxlength, filterThreshold);
}

inline IlcConstraint IlcParityConstraint(IloCP solver, IlcIntVarArray vars) {
  return IlcParityConstraint(solver, parity_filterLevel, parity_useBitxors, vars, parity_number, parity_minlength, parity_maxlength, parity_filterThreshold);
}



#endif

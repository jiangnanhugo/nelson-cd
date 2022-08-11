#ifndef XORS_CP
#define XORS_CP

#include <ilcp/cpext.h>

using namespace std;


static const int DEFAULT_FILTER_THRESHOLD = 4;
static const int WHEN_GRAPH_FILTERING_IS_BETTER = 5;



// Useful type definitions
typedef IlcRevBool* IlcRevBoolPtr;
//ILCARRAY(IlcRevBoolPtr)
//typedef IlcRevBoolPtrArray IlcRevBoolArray;


//*****************************************************************************
// The xor_cp constraint
//*****************************************************************************

class IlcXorCPConstraintI: public IlcConstraintI {
 private:
  const IloCP s;
  IlcIntVarArray vars;
  const int rhs;
  const int modulus_c;
  const int num_watched;

  int *watched;

  struct evaluate_t;
  evaluate_t evaluate(void);
  void filterGraphBased(const int num_active, const int target_value);
  void filterBruteForce(const int num_active, const int target_value);
  bool get_next_values(IlcIntArray values, IlcInt & value_sum) const;

 public:
  IlcXorCPConstraintI(const IloCP solver, const IlcIntVarArray Vars, const IlcInt Rhs, const IlcInt Mod_c, const IlcInt FilterThreshold);
  ~IlcXorCPConstraintI();

  virtual void post();
  virtual void propagate();
  void postCst();

  void print(ostream & os = cout) const;
  friend ostream & operator << (ostream & os, const IlcXorCPConstraintI cons);

};

IlcConstraint IlcXorCPConstraint(IloCP solver, IlcIntVarArray vars, IlcInt rhs, IlcInt modulus_c, IlcInt filter_threshold = DEFAULT_FILTER_THRESHOLD);


struct IlcXorCPConstraintI::evaluate_t {
  int numActive;
  int valueLHS;
  evaluate_t() : numActive(0), valueLHS(0) {}
};


#endif

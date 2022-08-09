#ifndef XORS
#define XORS

#include <ilcp/cpext.h>

using namespace std;


//*****************************************************************************
// The xor constraint
//*****************************************************************************

typedef enum {even, odd} parity_t;


class IlcXorConstraintI: public IlcConstraintI {
 private:
  const IloCP s;
  IlcIntVarArray vars;
  const parity_t parity;

  int watchedA, watchedB;

  struct evaluate_t;
  evaluate_t evaluate(void);

 public:
  IlcXorConstraintI(const IloCP solver, const IlcIntVarArray Vars, const IlcBool IsEven);
  ~IlcXorConstraintI();

  virtual void post();
  virtual void propagate();
  void postCst();

  void print(ostream & os = cout) const;
  friend ostream & operator << (ostream & os, const IlcXorConstraintI cons);

};

IlcConstraint IlcXorConstraint(IloCP solver, IlcIntVarArray vars, IlcBool isEven);


struct IlcXorConstraintI::evaluate_t {
  int numActive;
  parity_t valueLHS;
  evaluate_t() : numActive(0), valueLHS(even) {}
};


//----------- Inline functions ------------//
inline parity_t operator ^ (const parity_t p1, const parity_t p2) {
  return (p1 == p2) ? even : odd;
}
inline parity_t operator ^= (parity_t & p1, const parity_t p2) {
  return p1 = (p1 ^ p2);
}


//---------------- Other ------------------//
ostream & operator << (ostream & os, const parity_t parity);




#endif

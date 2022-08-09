#include "xors.h"

#define DBG(x)       // use this to not print debug info
//#define DBG(x) x     // use this to print debug info

#define NDBG(x) x




//-----------------------------------------------------------------------------
IlcXorConstraintI::IlcXorConstraintI(const IloCP solver, const IlcIntVarArray Vars, const IlcBool IsEven)
  : IlcConstraintI(solver), s(solver), vars(Vars), parity(IsEven?even:odd), watchedA(0), watchedB(1) {

}


//-----------------------------------------------------------------------------
IlcXorConstraintI::~IlcXorConstraintI(){
  // delete created memory 
}

//-----------------------------------------------------------------------------
ILCCTDEMON0(IlcXorPostDemon, IlcXorConstraintI, postCst);

void IlcXorConstraintI::postCst() {
  this->push(1);
}

void IlcXorConstraintI::post(){
  for (int i=0; i<vars.getSize(); i++)
    vars[i].whenDomain( IlcXorPostDemon(getSolver(), this) );
}

//-----------------------------------------------------------------------------
void IlcXorConstraintI::propagate()
{
  if (watchedA != watchedB 
      && !vars[watchedA].isBound()
      && !vars[watchedB].isBound())
    return;

  DBG(s.out() << "Propagating XorConstraint because of vars["
       << this->vars.getIndexValue() << ']' <<endl;);
  DBG(print());

  evaluate_t result = evaluate();
  if (result.numActive == 0) {
    if (result.valueLHS != parity)
      s.fail();
  }
  else if (result.numActive == 1) {
    DBG(s.out() << "   pruning domain value " 
         << (result.valueLHS==parity ? 1 : 0)
         << " for var " << watchedA << endl;);
    vars[watchedA].removeValue(result.valueLHS==parity ? 1 : 0);
  }
  
  DBG(s.out() << "Result:" << endl;);
  DBG(print(s.out()));
}

//-----------------------------------------------------------------------------
IlcXorConstraintI::evaluate_t IlcXorConstraintI::evaluate(void) {
  DBG(s.out() << "Watched variables " << watchedA << " and " << watchedB;);
  const int length = vars.getSize();
  const int startIdx = watchedA;

  evaluate_t result;
  int i = startIdx;
  do {
    if (!vars[i].isInDomain(0))
      result.valueLHS ^= odd;
    else if (vars[i].isInDomain(1)) {
      ++result.numActive;
      if (result.numActive == 1)
        watchedA = i;
      else if (result.numActive == 2) {
        watchedB = i;
        break;
      }
    }
    i = (i+1) % length;
  } while (i != startIdx);

  DBG(s.out() << "  changed to  variables " << watchedA << " and " << watchedB << endl;);
  return result;
}

//-----------------------------------------------------------------------------
void IlcXorConstraintI::print(ostream & os) const {
  if (vars.getSize() == 0) {
    if (parity == even)
      os << "Empty satisfied constraint (RHS == even)" << endl;
    else
      os << "Empty violated constraint (RHS == odd)" << endl;
    return;
  }
  for (int i=0; i<vars.getSize(); ++i) {
    if (i != 0)
      os << " + ";
    if (watchedA == i || watchedB == i)
      os << '*' << i << '*';
    else
      os << i;
    if (!vars[i].isInDomain(0))
      os << "(1)";
    if (!vars[i].isInDomain(1))
      os << "(0)";
  }
  os << " = " << parity << endl;
}

//-----------------------------------------------------------------------------
ostream & operator << (ostream & os, const IlcXorConstraintI cons) {
  cons.print();
  return os;
}

//-----------------------------------------------------------------------------
IlcConstraint IlcXorConstraint(IloCP solver, IlcIntVarArray vars, IlcBool isEven){
  
  return new (solver.getHeap()) IlcXorConstraintI(solver, vars, isEven);
}



////////////////////////////////////

ostream & operator << (ostream & os, const parity_t parity) {
  os << (parity == even ? "even" : "odd");
  return os;
}


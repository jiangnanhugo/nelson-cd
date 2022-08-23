#ifndef PARITY_WRAPPER
#define PARITY_WRAPPER


ILOCPCONSTRAINTWRAPPER6(IloParityConstraint, solver, IloInt, _filterLevel, IloBool, _useBitxors, IloIntVarArray, vars, IloInt, _number, IloInt, _minlength, IloInt, _maxlength){
  
  use(solver, vars);

  //return IlcParityConstraint(solver, solver.getInt(_filterLevel), solver.getIntVarArray(vars), solver.getInt(_number), solver.getInt(_minlength), solver.getInt(_maxlength));
  return IlcParityConstraint(solver, _filterLevel, _useBitxors, solver.getIntVarArray(vars), _number, _minlength, _maxlength);
}

ILOCPCONSTRAINTWRAPPER1(IloParityConstraint1, solver, IloIntVarArray, vars) {
  
  use(solver, vars);
  return IlcParityConstraint(solver, solver.getIntVarArray(vars));
}

#endif

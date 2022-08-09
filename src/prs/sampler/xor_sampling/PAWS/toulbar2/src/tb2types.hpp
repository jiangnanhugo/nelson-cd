/** \file tb2types.hpp
 *  \brief Macros, types, and globals.
 *
 * The main types are:
 * - ::Value : domain value
 * - ::Cost : cost value (exact type depends on compilation flag)
 * - ::Long : large integer (long long int)
 * - ::TProb : probability value (exact type depends on compilation flag)
 * - ::Double : large float (long double)
 * - ::String : string with extended character size (wide string) to encode tuples
 *
 * \note Compilation flag for Cost is: \c INT_COST (int), \c LONGLONG_COST (long long), or \c PARETOPAIR_COST (see ::ParetoPair)
 * \warning \c PARETOPAIR_COST is fragile.
 * \note Compilation flag for TProb is: \c DOUBLE_PROB or \c LONGDOUBLE_PROB
 * \note Compilation flag for String is: \c WIDE_STRING or nothing (usual C++ string)
 */

#ifndef TB2TYPES_HPP_
#define TB2TYPES_HPP_

//#define INT_COST
//#define LONGLONG_COST
//#define PARETOPAIR_COST

//#define DOUBLE_PROB
//#define LONGDOUBLE_PROB

// uncomment if using large enumerated domains with BTD or in nary cost functions
//#define WIDE_STRING

#include "tb2utils.hpp"
#include "tb2integer.hpp"

/// Domain value (can be positive or negative integers)
typedef int Value;
/// Maximum domain value
const Value MAX_VAL = (INT_MAX / 2);
/// Minimum domain value
const Value MIN_VAL = -(INT_MAX / 2);
/// Maximum domain size
/// \deprecated Should use WCSP::getMaxDomainSize instead.
const Value MAX_DOMAIN_SIZE = 2000;

#ifdef INT_COST
const bool PARTIALORDER = false;
typedef int Cost;
const Cost MIN_COST = 0;
const Cost UNIT_COST = 1;
const Cost SMALL_COST = 1;
const Cost MEDIUM_COST = 3;
const Cost LARGE_COST = 100;
const Cost MAX_COST = ((INT_MAX / 2) / MEDIUM_COST);
inline Cost MIN(Cost a, Cost b) {return min(a,b);}
inline Cost MAX(Cost a, Cost b) {return max(a,b);}
inline Cost GLB(Cost a, Cost b) {return MIN(a,b);}
inline Cost LUB(Cost a, Cost b) {return MAX(a,b);}
inline bool GLB(Cost *a, Cost b) {if (b < *a) {*a = b; return true;} else return false;}
inline bool LUB(Cost *a, Cost b) {if (b > *a) {*a = b; return true;} else return false;}
inline bool GLBTEST(Cost a, Cost b) {return (b < a);}
inline bool LUBTEST(Cost a, Cost b) {return (b > a);}
inline bool DACTEST(Cost a, Cost b) {return (a==0 && b>0);}
inline bool SUPPORTTEST(Cost a, Cost b) {return false;}
inline bool SUPPORTTEST(Cost a) {return false;}
inline bool CUT(Cost lb, Cost ub) {return lb >= ub;}
inline bool CSP(Cost lb, Cost ub) {return (ub - lb) <= 1;}
inline void initCosts(Cost ub) {}
#endif

#ifdef LONGLONG_COST
const bool PARTIALORDER = false;
typedef Long Cost;
const Cost MIN_COST = 0;
const Cost UNIT_COST = 1;
const Cost SMALL_COST = 1;
const Cost MEDIUM_COST = 3;
const Cost LARGE_COST = 100;
const Cost MAX_COST = ((LONGLONG_MAX / 2) / MEDIUM_COST);
inline Cost MIN(Cost a, Cost b) {
	return min(a, b);
}
inline Cost MAX(Cost a, Cost b) {
	return max(a, b);
}
inline Cost GLB(Cost a, Cost b) {
	return MIN(a, b);
}
inline Cost LUB(Cost a, Cost b) {
	return MAX(a, b);
}
inline bool GLB(Cost *a, Cost b) {
	if (b < *a) {
		*a = b;
		return true;
	} else
		return false;
}
inline bool LUB(Cost *a, Cost b) {
	if (b > *a) {
		*a = b;
		return true;
	} else
		return false;
}
inline bool GLBTEST(Cost a, Cost b) {
	return (b < a);
}
inline bool LUBTEST(Cost a, Cost b) {
	return (b > a);
}
inline bool DACTEST(Cost a, Cost b) {
	return (a == 0 && b > 0);
}
inline bool SUPPORTTEST(Cost a, Cost b) {
	return false;
}
inline bool SUPPORTTEST(Cost a) {
	return false;
}
inline bool CUT(Cost lb, Cost ub) {
	return lb >= ub;
}
inline bool CSP(Cost lb, Cost ub) {
	return (ub - lb) <= 1;
}
inline void initCosts(Cost ub) {
}
#endif

#ifdef PARETOPAIR_COST
const bool PARTIALORDER = true;
#include "tb2paretopair.hpp"
typedef ParetoPair Cost;
const Cost MIN_COST = PARETOPAIR_MIN;
const Cost UNIT_COST = PARETOPAIR_1;
const Cost SMALL_COST = PARETOPAIR_1;
const Cost MEDIUM_COST = PARETOPAIR_3;
const Cost LARGE_COST = PARETOPAIR_100;
const Cost MAX_COST = PARETOPAIR_MAX;
#endif

#ifdef DOUBLE_PROB
typedef double TProb;
#endif

#ifdef LONGDOUBLE_PROB
typedef Double TProb;
#endif

const int STORE_SIZE = 16;
#define INTEGERBITS (8*sizeof(Cost)-2)

const int MAX_ELIM_BIN = 1000000000;
const int MAX_ARITY = 1000;

typedef map<int, int> TSCOPE;
typedef map<int, Value> TAssign;

#ifdef NARYCHAR
#define CHAR_FIRST '0'
#else
#define CHAR_FIRST 1
#endif

/*
 * Global variables encapsulated as static members
 *
 */

typedef void (*externalevent)(int wcspId, int varIndex, Value value, void *solver);
typedef void (*externalcostevent)(int wcspId, int varIndex, Cost cost, void *solver);
typedef void (*externalsolution)(int wcspId, void *solver);

typedef enum {
	ELIM_NONE = 0, MAX_CARD = 1, MIN_FILL = 2, MIN_DEGREE = 3, ELIM_MAX
} ElimOrderType;

class Pedigree;
class Haplotype;
class BEP;

typedef enum {
	LC_NC = 0,
	LC_SNIC = 0,
	LC_AC = 1,
	LC_DAC = 2,
	LC_FDAC = 3,
	LC_EDAC = 4,
	LC_THEMAX
} LcLevelType;

///contains all global variables (mainly solver's command-line options)
class ToulBar2 {
protected:
	virtual ~ToulBar2() = 0; // Trick to avoid any instantiation of ToulBar2
public:
	static string version;
	static int verbose;
	static int debug;
	static bool showSolutions;
	static bool writeSolution;
	static bool allSolutions;
	static int dumpWCSP;
	static bool approximateCountingBTD;
	static bool binaryBranching;
	static bool dichotomicBranching;
	static unsigned int dichotomicBranchingSize;
	static bool sortDomains;
	static int elimDegree;
	static int elimDegree_preprocessing;
	static int elimDegree_;
	static int elimDegree_preprocessing_;
	static int elimSpaceMaxMB;
	static int minsumDiffusion;
	static int preprocessTernaryRPC;
	static int preprocessFunctional;
	static bool costfuncSeparate;
	static int preprocessNary;
	static bool QueueComplexity;
	static bool Static_variable_ordering;// flag for static variable ordering during search (dynamic ordering is default value)
	static bool lastConflict;
	static int weightedDegree;
    static int nbDecisionVars;
	static int lds;
	static bool limited;
	static Long restart;
	static externalevent setvalue;
	static externalevent setmin;
	static externalevent setmax;
	static externalevent removevalue;
	static externalcostevent setminobj;
	static externalsolution newsolution;
	static Pedigree *pedigree;
	static Haplotype *haplotype;
	static string map_file;
	static bool bayesian;
	static bool uai;
	static int resolution;
	static TProb errorg;
	static TProb NormFactor;
	static int foundersprob_class;
	static vector<TProb> allelefreqdistrib;
	static bool consecutiveAllele;
	static bool generation;
	static int pedigreeCorrectionMode;
	static int pedigreePenalty;
	static int vac;
	static Cost costThreshold;
	static Cost costThresholdPre;
	static Cost costMultiplier;
	static Cost relaxThreshold;
	static ElimOrderType elimOrderType;
	static bool singletonConsistency;
	static bool vacValueHeuristic;
	static BEP *bep;
	static LcLevelType LcLevel;
	static bool wcnf;
	static bool qpbo;

	static char* varOrder;
	static int btdMode;
	static int btdSubTree;
	static int btdRootCluster;

	static bool xmlflag;
	static TProb markov_log;
	static string evidence_file;
	static ofstream solution_file;
    static bool uai_firstoutput;

	static double startCpuTime;

	static int splitClusterMaxSize;
	static bool boostingBTD;
	static int maxSeparatorSize;
	static int minProperVarSize;

	static int smallSeparatorSize;

	static bool isZ;
	static TProb logZ;
    static Cost negCost;
	static int Berge_Dec; // flag for berge acyclic decomposition
	static int nbvar; // initial number of variable (read in the file) 

	
};

/*
 * Backtrack exception
 *
 */
class Contradiction {
public:
	Contradiction() {
		if (ToulBar2::verbose >= 2)
			cout << "... contradiction!" << endl;
	}
};
#ifdef ILOGLUE
//extern IloSolver IlogSolver;
extern IloCP IlogSolver;
#define THROWCONTRADICTION ({if (ToulBar2::verbose >= 2) cout << "... contradiction!" << endl; if (ToulBar2::weightedDegree) conflict(); IlogSolver.fail(0);})
#else

#define THROWCONTRADICTION {if (ToulBar2::weightedDegree) conflict(); throw Contradiction();}
#endif

/*
 * Internal classes and basic data structures used everywhere
 *
 */

class Store;
class Domain;
class Variable;
class IntervalVariable;
class EnumeratedVariable;
class Constraint;
class WCSP;
class Solver;
class Cluster;
class Separator;
class TreeDecomposition;

struct ValueCost {
	Value value;
	Cost cost;
};

struct ConstraintLink {
	Constraint *constr;
	int scopeIndex;
};

class WCSPLink {
public:
	WCSP * const wcsp;
	int wcspIndex;
	WCSPLink(WCSP *w, int index) :
		wcsp(w), wcspIndex(index) {
	}
};

#endif /*TB2TYPES_HPP_*/

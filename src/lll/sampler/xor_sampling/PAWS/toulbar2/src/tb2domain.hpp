/** \file tb2domain.hpp
 *  \brief Storable enumerated domain.
 * 
 */
 
#ifndef TB2DOMAIN_HPP_
#define TB2DOMAIN_HPP_

#include "tb2btlist.hpp"

extern int cmpValue(const void *v1, const void *v2);

class Domain : public BTList<Value>
{
    const int initSize;
    const Value distanceToZero;
    DLink<Value> *all;

    void init(Value inf, Value sup);
    
    // make it private because we don't want copy nor assignment
    Domain(const Domain &s);
    Domain& operator=(const Domain &s);
        
public:
    typedef BTList<Value>::iterator iterator;
    
    Domain(Value inf, Value sup, StoreStack<BTList<Value>, DLink<Value> *> *s);
    
    Domain(Value *d, int dsize, StoreStack<BTList<Value>, DLink<Value> *> *s);
    
    ~Domain() {if (initSize >= 1) delete[] all;}
        
    int getInitSize() const {return initSize;}
    int toIndex(Value v) const {return v - distanceToZero;}
    Value toValue(int idx) const {return idx + distanceToZero;}
    int toCurrentIndex(Value v) {
	  assert(canbe(v));
	  int pos=0; 
	  for (iterator iter = begin(); iter != end(); ++iter) {
		if (*iter == v) return pos;
		pos++;
	  }
	  cerr << "Bad (removed) value given as argument of toCurrentIndex function!" << endl;
	  exit(EXIT_FAILURE);
	  return -1;
	}

    bool canbe(Value v) const {return !all[toIndex(v)].removed;}
    bool cannotbe(Value v) const {return all[toIndex(v)].removed;}

    void erase(Value v) {BTList<Value>::erase(&all[toIndex(v)], true);}

    int increase(Value v) {
        iterator newInf = lower_bound(v);
        assert(canbe(*newInf));
        for (iterator iter = begin(); iter != newInf; ++iter) {
            erase(*iter);
        }
        return *newInf;
    }
    int decrease(Value v) {
        iterator newSup = upper_bound(v);
        assert(canbe(*newSup));
        for (iterator iter = rbegin(); iter != newSup; --iter) {
            erase(*iter);
        }
        return *newSup;
    }
    
    //Finds the first available element whose value is greater or equal to v
    iterator lower_bound(Value v) {
        iterator iter(&all[toIndex(v)]);
        if (cannotbe(v)) {
            ++iter;
        }
        return iter;
    }
    
    //Finds the first available element whose value is lower or equal to v
    iterator upper_bound(Value v) {
        iterator iter(&all[toIndex(v)]);
        if (cannotbe(v)) {
            --iter;
        }
        return iter;
    }
    
    friend ostream& operator<<(ostream& os, Domain &l);
};

#endif /*TB2DOMAIN_HPP_*/

"""
notations on encoding:
- Variables: encoded as numbers 0 to n-1;
- Literal l: v is 2*v and ~v is 2*v+1. so the foremost bit of a literal encodes whether it is negated or not.
           This can be tested simply checking if l&1 is 0 or 1.
- To negate a literal, we jsut have to toggle the foremost bit. THis can be done by XOR with 1:
  The negation of l is: l XOR 1.
- To get a lieral's variable, we just need to shift to the right. THis can be done with l >>1.

E.g., variable b is encoded with number 3. The literal b is encoded as: 2*3 = 6 and ~b as 2*3+1=7
"""

class SAT_Instance(object):
    def __init__(self) -> None:
        self.variables = []
        self.variable_table = dict()
        self.clauses = []
    
    def parse_and_add_clause(self, line):
        clause = []
        for literal in line.split():
            if literal.startswith("~"):
                negated = 1
            else:
                negated = 0
            variable = literal[negated:]
            if variable not in self.variable_table:
                self.variable_table[variable] = len(self.variables)
                self.variables.append(variable)
            encoded_literal = self.variable_table[variable] << 1 | negated
            clause.append(encoded_literal)
        self.clauses.append(tuple(set(clause)))
    
    @classmethod
    def from_file(cls, file):
        instance = cls()
        for line in file:
            line = line.strip()
            if len(line) > 0 and not line.startswith("#"):
                instance.parse_and_add_clause(line)
        return instance
    
    def literal_to_str(self, literal):
        if literal & 1:
            s = '~'
        else:
            s =''
        return s + self.variables[literal >> 1]
    
    def clause_to_str(self, clause):
        return ' '.join(self.literal_to_str(l) for l in clause)
    
    def assignment_to_str(self, assignment, brief=False, starting_with=""):
        literals = []
        for a, v in ((a,v) for a,v in zip(assignment, self.variables) if v.startswith(starting_with)):
            if a == 0 and not brief:
                literals.append("~"+ v)
            elif a:
                literals.append(v)
        return " ".join(literals)
             
#include<bits/stdc++.h>

using namespace std;
typedef vector<int> vi;

const numvars = 11500;
const int k=10;
const double e = exp(1);
const int apprearances = (int) (pow(2,k)/e/k);
const clauses = numvars / 10 *37;
vi ans[clauses];
int perm[clauses];

int main(){
    for (int i = 0; i < clauses;i++) perm[i]=i;
    int curvar = 0;
    for(int i=0; i < k; i++){
        for(int j=clauses - 1; j > 0; j--){
            int idx = rand() % (j+1);
            swap(perm[j], perm[idx]);
        }
        for(int j=0;j< clauses / apprearances;j++){
            for (int of=0;of< apprearances;of++){
                ans[perm[j*apprearances + of]].push_back(curvar);
            }
            curvar++;
        }
    }
    printf("number of variables:%d\n", curvar);
    for(int i=0;i<clauses;i++){
        for(int j:ans[i]){
            if(rand() &1) printf("~");
            printf("%d ", j);
        }
        printf("\n");
    }
}
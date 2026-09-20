#include <quadmath.h>
#include <iostream>
int main(){__float128 x=strtoflt128("2",nullptr);char s[128];quadmath_snprintf(s,sizeof(s),"%.36Qg",logq(x));std::cout<<s<<"\n";}

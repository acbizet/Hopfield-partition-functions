// Matched full rank-five tensor evaluators. No spin expansion in this program.
// g++ -O3 -std=c++17 evaluate.cpp -lquadmath -o evaluate
#include <quadmath.h>
#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

constexpr int N=8,R=5;
template<class T> T parse(const std::string&s);
template<> double parse<double>(const std::string&s){return std::strtod(s.c_str(),nullptr);}
template<> long double parse<long double>(const std::string&s){return std::strtold(s.c_str(),nullptr);}
template<> __float128 parse<__float128>(const std::string&s){return strtoflt128(s.c_str(),nullptr);}
inline double ex(double x){return std::exp(x);} inline long double ex(long double x){return std::exp(x);} inline __float128 ex(__float128 x){return expq(x);}
inline double lg(double x){return std::log(x);} inline long double lg(long double x){return std::log(x);} inline __float128 lg(__float128 x){return logq(x);}
inline double sq(double x){return std::sqrt(x);} inline long double sq(long double x){return std::sqrt(x);} inline __float128 sq(__float128 x){return sqrtq(x);}
template<class T> std::string repr(T x){std::ostringstream s;s<<std::setprecision(25)<<x;return s.str();}
template<> std::string repr(__float128 x){char s[256];quadmath_snprintf(s,sizeof(s),"%.38Qg",x);return s;}

struct Raw {
 std::string beta; std::array<std::string,N>b; std::array<std::array<int,R>,N>w;
 std::vector<std::pair<std::string,std::string>> rule;
};
template<class T> struct Evaluator {
 const Raw&raw; T beta;std::array<T,N>b;std::array<std::vector<T>,R> weights;
 std::array<std::vector<std::array<T,N>>,R> factor;unsigned long long points=1;
 Evaluator(const Raw&r):raw(r){beta=parse<T>(r.beta);for(int i=0;i<N;i++)b[i]=parse<T>(r.b[i]);}
 template<int depth> T sum(const std::array<T,N>&field) const {
  T total=0;
  for(size_t k=0;k<weights[depth].size();k++){
   if constexpr(depth==R-1){
    T value=1;
    for(int i=0;i<N;i++){T e=field[i]*factor[depth][k][i];value*=e+T(1)/e;}
    total+=weights[depth][k]*value;
   }else{
    std::array<T,N> child;
    for(int i=0;i<N;i++)child[i]=field[i]*factor[depth][k][i];
    total+=weights[depth][k]*sum<depth+1>(child);
   }
  }
  return total;
 }
 T run(const std::string&method,int order){
  points=1;
  T pi=parse<T>("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068");
  T G=1;
  if(method=="positive"){
   T a=2*beta/N;
   for(int j=1;j<10000;j++){
    G+=2*ex(-a*T(j*j));
    T tail=2*ex(-a*T((j+1)*(j+1)))/(1-ex(-a*T(2*j+3)));
    if(tail<parse<T>("1e-42"))break;
   }
  }
  for(int a=0;a<R;a++){
   std::vector<T>x;weights[a].clear();factor[a].clear();
   if(method=="positive"){
    int p=0;for(int i=0;i<N;i++)p+=raw.w[i][a];p=(p%2+2)%2;
    for(int k=-order;k<=order-p;k++){
     T m=T(p+2*k)/N;x.push_back(m);weights[a].push_back(ex(-N*beta*m*m/2)/G);
    }
   }else{
    T scale=sq(T(2)/(N*beta));T norm=sq(pi);
    for(const auto&z:raw.rule){x.push_back(parse<T>(z.first)*scale);weights[a].push_back(parse<T>(z.second)/norm);}
   }
   factor[a].resize(x.size());points*=x.size();
   for(size_t k=0;k<x.size();k++)for(int i=0;i<N;i++)factor[a][k][i]=raw.w[i][a]==0?T(1):ex(beta*T(raw.w[i][a])*x[k]);
  }
  std::array<T,N>field;for(int i=0;i<N;i++)field[i]=ex(beta*b[i]);
  return lg(sum<0>(field));
 }
};
template<class T> void measure(const Raw&raw,const std::string&method,int order,int repeats){
 Evaluator<T>e(raw);T value=e.run(method,order);std::vector<double>times;std::vector<std::string>values;
 volatile T sink=value;
 for(int j=0;j<repeats;j++){
  auto t=std::chrono::steady_clock::now();value=e.run(method,order);sink=value;
  double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();times.push_back(dt);values.push_back(repr(value));
 }
 auto sorted=times;std::sort(sorted.begin(),sorted.end());
 std::cout<<std::setprecision(15)<<"{\"logZ\":\""<<repr(value)<<"\",\"median_seconds\":"<<sorted[sorted.size()/2]<<",\"points\":"<<e.points<<",\"samples_seconds\":[";
 for(size_t j=0;j<times.size();j++){if(j)std::cout<<",";std::cout<<times[j];}
 std::cout<<"]}"<<std::endl;
}
int main(int argc,char**argv){
 if(argc!=7){std::cerr<<"evaluate input rule-or-dash precision method order repeats\n";return 2;}
 Raw r;std::ifstream f(argv[1]);if(!f)return 3;f>>r.beta;for(auto&b:r.b)f>>b;for(auto&row:r.w)for(int&w:row)f>>w;
 std::string method=argv[4];int order=std::stoi(argv[5]),repeats=std::stoi(argv[6]);
 if(method!="positive"){
  std::ifstream g(argv[2]);std::string x,w;while(g>>x>>w)r.rule.emplace_back(x,w);
  if(int(r.rule.size())!=order)return 4;
 }
 std::string precision=argv[3];
 if(precision=="double")measure<double>(r,method,order,repeats);
 else if(precision=="longdouble")measure<long double>(r,method,order,repeats);
 else if(precision=="quad")measure<__float128>(r,method,order,repeats);
 else return 5;
}

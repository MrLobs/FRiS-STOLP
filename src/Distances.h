#include <boost/make_shared.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/shared_ptr.hpp>

using namespace std;

namespace fris{

enum metric {euclidean = 0, manhattan = 1, sq_euclidean = 2};

typedef boost::numeric::ublas::matrix<double> bmatrix;
typedef boost::numeric::ublas::vector<double> bvector;

class Distances {
 protected:
  boost::shared_ptr<const bmatrix> pX;
  metric m;
  double distance(const bvector& x) {
      switch (m){
      case euclidean:
          return norm_2(x);
      case manhattan:
          return norm_1(x);
      case sq_euclidean:
          double n = 0;
          for ( auto i : x ){
              n += i*i;
          }
          return n;
      }
  };

 public:
  Distances(metric m_) : m(m_) {};
  virtual void init(const boost::shared_ptr<const bmatrix>& A) = 0;
  virtual double get(int i, int j) = 0;
  double get(const bvector& x, int i) { return distance(row(*pX, i) - x); }
  virtual ~Distances(){};
};

class Distances_v1 : public Distances {
  bmatrix D;

 public:
  Distances_v1(metric m_) : Distances(m_) {};
  ~Distances_v1(){};
  void init(const boost::shared_ptr<const bmatrix>& A) {
    pX = A;
    D.clear();
    D.resize(pX->size1(), pX->size1());
    for (int i = 0; i < pX->size1(); ++i) {
      for (int j = i; j < pX->size1(); ++j) {
        D(i, j) = distance(row(*pX, i) - row(*pX, j));
        D(j, i) = D(i, j);
      }
    }
  }
  double get(int i, int j) { return D(i, j); };
};

class Distances_v2 : public Distances {
 public:
  Distances_v2(metric m_) : Distances(m_) {};
  ~Distances_v2(){};
  void init(const boost::shared_ptr<const bmatrix>& A) { pX = A; }
  double get(int i, int j) { return distance(row(*pX, i) - row(*pX, j)); }
};
}

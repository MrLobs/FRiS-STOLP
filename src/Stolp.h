#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/shared_ptr.hpp>
#include <unordered_set>
#include "Distances.h"

using namespace std;

namespace fris{

class FRiS_Stolp {
  boost::shared_ptr<Distances> Dist;
  unordered_set<int> e0, e1;
  boost::shared_ptr<const bmatrix> X;
  vector<char> Y;
  double threshold;
  int size;

  template <typename T, typename T2>  // T = int for fit(), T = bvector for predict()
  int nearestNeighbour(const T& i, const T2& x); //T2 vector or unordered set
  template <typename T2>
  int findEtalon(const vector<int>& x, const T2& e,
                 const vector<int>& x_);
  template <typename T>
  double FRiS(const T& i, int j, int k);

 public:
  FRiS_Stolp(double threshold_, int v, metric m = euclidean) : threshold(threshold_) {
    if (v == 1) {
      Dist = boost::shared_ptr<Distances>(new Distances_v1(m));
    } else {
      Dist = boost::shared_ptr<Distances>(new Distances_v2(m));
    }
  };
  void fit(const bmatrix& X_train, const vector<char>& Y_train);
  vector<char> predict(const bmatrix& X_test);
  vector<double> predict_proba(const bmatrix& X_test);
  vector<char> isEtalon();
};
}

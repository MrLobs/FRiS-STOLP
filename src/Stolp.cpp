#include "Stolp.h"
#include <algorithm>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>

using namespace std;

namespace fris {

void FRiS_Stolp::fit(const bmatrix& X_train, const vector<char>& Y_train) {
  X = boost::make_shared<const bmatrix>(X_train);
  Y = Y_train;
  size = Y.size();
  e0.clear();
  e1.clear();
  Dist->init(X);

  vector<int> X0, X1, X_;
  unordered_set<int> e0_, e1_;
  int e0_cur, e1_cur;
  for (int i = 0; i < size; ++i) {
    if (Y[i]) {
      X1.push_back(i);
    } else
      X0.push_back(i);
  }

  assert(!X0.empty());
  assert(!X1.empty());
  bool isENew = true;
  bool isStolpElements0 = false;
  bool isStolpElements1 = false;
  while (isENew && !(isStolpElements0 && isStolpElements1)) {
    e0_ = e0;
    e1_ = e1;
    e0_.insert(findEtalon(X0, X1, X1));
    e1_.insert(findEtalon(X1, X0, X0));
    e0_cur = findEtalon(X0, e1_, X1);
    e1_cur = findEtalon(X1, e0_, X0);
    isENew = (!e0.count(e0_cur) || !e1.count(e1_cur));
    e0.insert(e0_cur);
    e1.insert(e1_cur);

    X_.clear();
    for (auto i : X0) {
      if ( e0.count(i) ||
          FRiS(i, nearestNeighbour(i, e0), nearestNeighbour(i, e1)) <
          threshold)
        X_.push_back(i);
    }
    swap(X0, X_);

    X_.clear();
    for (auto i : X1) {
      if ( e1.count(i) ||
          FRiS(i, nearestNeighbour(i, e1), nearestNeighbour(i, e0)) <
          threshold)
        X_.push_back(i);
    }
    swap(X1, X_);
  }
}

template<typename T2>
int FRiS_Stolp::findEtalon(const vector<int>& X, const T2& e,
                           const vector<int>& X_) {
  if (X.size() == 1) return X[0];
  double sum, f, s, s_max = std::numeric_limits<double>::min();
  int i_max;
  for (auto i : X) {
    // Defensibility
    sum = 0;
    for (auto j : X) {
      f = FRiS(j, i, nearestNeighbour(j, e));
      if (j != i) sum += f;
    }
    s = sum / (X.size() - 1);
    // Tolerance
    sum = 0;
    for (auto j : X_) {
      f = FRiS(j, nearestNeighbour(j, e), i);
      sum += f;
    }
    s = (s + sum / X_.size()) / 2;

    if (s > s_max) {
      s_max = s;
      i_max = i;
    }
  }
  return i_max;
}

template <typename T, typename T2>
int FRiS_Stolp::nearestNeighbour(const T& i, const T2& X) {
  return *std::min_element(X.begin(), X.end(), [&](int j1, int j2){
      return Dist->get(i, j1) < Dist->get(i, j2);
  });
}

template <typename T>
double FRiS_Stolp::FRiS(const T& i, int j, int k) {
  return (Dist->get(i, k) - Dist->get(i, j)) /
         (Dist->get(i, j) + Dist->get(i, k));
}

vector<char> FRiS_Stolp::predict(const bmatrix& X_test) {
  vector<char> Y;
  bvector x;
  double d0, d1;
  for (int i = 0; i < X_test.size1(); ++i) {
    x = row(X_test, i);
    d0 = Dist->get(x, nearestNeighbour(x, e0));
    d1 = Dist->get(x, nearestNeighbour(x, e1));
    Y.push_back(d0 > d1 ? 1 : 0);
  }
  return Y;
}

vector<double> FRiS_Stolp::predict_proba(const bmatrix& X_test) {
  vector<double> Y;
  bvector x;
  double d0, d1, f;
  for (int i = 0; i < X_test.size1(); ++i) {
    x = row(X_test, i);
    f = FRiS(x, nearestNeighbour(x, e1), nearestNeighbour(x, e0));
    Y.push_back((f + 1) / 2);  // <-TODO
  }
  return Y;
}

vector<char> FRiS_Stolp::isEtalon() {
  vector<char> Y(size);
  for (auto i : e0) Y[i] = 1;
  for (auto i : e1) Y[i] = 1;
  return Y;
}

}

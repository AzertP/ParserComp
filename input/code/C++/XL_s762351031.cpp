#define _CRT_SECURE_NO_WARNINGS
#include <bits/stdc++.h>
using namespace std;
//using ll=long long;
const double EPS = 1e-10;
inline bool equals(double a, double b) { return fabs(a - b) < EPS; }
const double PI = 3.141592653589793238;
class Point {
public:
	double x, y;
	Point(double x, double y) :x(x), y(y) {}
	Point() {}
	Point operator +(const Point &p) const{ return Point(x + p.x, y + p.y); };
	Point operator -(const Point &p) const{ return Point(x - p.x, y - p.y); }
	Point operator *(double k) { return Point(x*k, y*k); }
	Point operator /(double k) { return Point(x / k, y / k); }
	bool operator <(const Point &p)const {
		return (x != p.x) ? x < p.x : y < p.y;
	}
	bool operator ==(const Point &p)const{
		return fabs(x - p.x) < EPS && fabs(y - p.y) < EPS;
	}
	void show() { printf("%.10lf %.10lf", x, y); }
};

using Vector = Point;
inline double norm(Vector a) {
	return a.x*a.x + a.y*a.y;
}
double absv(Vector a) {
	return sqrt(norm(a));
}
inline double dot(Vector a, Vector b) {
	return a.x*b.x + a.y*b.y;
}
inline double cross(Vector a, Vector b) {
	return a.x*b.y - a.y*b.x;
}
struct Segment {
	Point p1, p2;
};

using Line = Segment;
class Circle {
public:
	Point c;
	double r;
	Circle(Point c = Point(), double r = 0.0) :c(c), r(r) {}
};
using Polygon = vector<Point>; 
Point project(Segment s, Point p) {
	Vector base = s.p2 - s.p1;
	double r = dot(p - s.p1, base) / norm(base);
	return s.p1 + base*r;
}
Point reflect(Segment s, Point p) {
	return p + (project(s, p) - p)*2.0;
}
double getDistanceLP(Line, Point);
double getDistance(Point, Point);
static const int COUNTER_CLOCKWISE = 1;
static const int CLOCKWISE = -1;
static const int ONLINE_BACK = 2;
static const int ONLINE_FRONT = -2;
static const int ON_SEGMENT = 0;

int ccw(Point p0, Point p1, Point p2) {
	Vector a = p1 - p0;
	Vector b = p2 - p0;
	if (cross(a, b) > EPS) return COUNTER_CLOCKWISE;
	if (cross(a, b) < -EPS) return CLOCKWISE;
	if (dot(a, b) < -EPS) return ONLINE_BACK;
	if (norm(a) < norm(b)) return ONLINE_FRONT;
	return ON_SEGMENT;
}
bool intersect(Point p1, Point p2, Point p3, Point p4) {
	return (ccw(p1, p2, p3)*ccw(p1, p2, p4) <= 0 && ccw(p3, p4, p1)*ccw(p3, p4, p2) <= 0);
}

bool intersect(Segment s1, Segment s2) {
	return intersect(s1.p1, s1.p2, s2.p1, s2.p2);
}
bool intersect(Circle c1, Circle c2) {
	return c1.r + c2.r >= getDistance(c1.c, c2.c);
}

bool intersect(Circle c, Line l) {
	double d = getDistanceLP(l, c.c);
	return d <= c.r;
}

double getDistance(Point a, Point b) {
	return absv(a - b);
}

double getDistanceLP(Line l, Point p) {
	return abs(cross(l.p2 - l.p1, p - l.p1) / absv(l.p2 - l.p1));
}

double getDistanceSP(Segment s, Point p) {
	if (dot(s.p2 - s.p1, p - s.p1) < 0.0) return absv(p - s.p1);
	if (dot(s.p1 - s.p2, p - s.p2) < 0.0) return absv(p - s.p2);
	return getDistanceLP(s, p);
}

double getDistance(Segment s1,Segment s2){
	if (intersect(s1, s2)) return 0.0;
	return min(min(getDistanceSP(s1, s2.p1), getDistanceSP(s1, s2.p2)), min(getDistanceSP(s2, s1.p1), getDistanceSP(s2, s1.p2)));
}

Point getCrossPoint(Segment s1, Segment s2) {
	Vector base = s2.p2 - s2.p1;
	double d1 = abs(cross(base, s1.p1 - s2.p1));
	double d2 = abs(cross(base, s1.p2 - s2.p1));
	return s1.p1 + (s1.p2 - s1.p1)*(d1 / (d1 + d2));
}
pair<Point, Point> getCrossPoints(Circle c, Line l) {
	assert(intersect(c, l));
	Point pr = project(l, c.c);
	Vector lv = l.p2 - l.p1;
	Vector le = lv / absv(lv);
	double base = sqrt(c.r*c.r - norm(c.c - pr));
	return make_pair(pr + le*base, pr - le*base);
}
double arg(Vector p) {
	return atan2(p.y, p.x);
}
Vector polar(double a, double r) {
	return Point(a * cos(r), a * sin(r));
}

pair<Point, Point> getCrossPoints(Circle c1, Circle c2) {
	assert(intersect(c1, c2));
	double d = getDistance(c1.c, c2.c);
	double a = acos((c1.r*c1.r + d*d - c2.r*c2.r) / (2 * c1.r*d));
	double t = arg(c2.c - c1.c);
	return make_pair(c1.c + polar(c1.r, a + t), c1.c + polar(c1.r, t - a));
}

int contains(Polygon g, Point p) {
	int n = g.size();
	bool x = false;
	for (int i = 0; i < n; i++) {
		Point a = g[i] - p; Point b = g[(i + 1) % n] - p;
		if (abs(cross(a, b)) < EPS && dot(a, b) <= 0) return 1;
		if (a.y > b.y) swap(a, b);
		if (a.y < EPS && EPS<b.y && cross(a, b)>EPS) x = !x;
	}
	return (x ? 2 : 0);
	
}

//Polygon andrewScan(Polygon s) {
//	Polygon u, l;
//	if (s.size() < 3) return s;
//	sort(s.begin(), s.end());
//	u.push_back(s[0]);
//	u.push_back(s[1]);
//	l.push_back(s[s.size() - 1]);
//	l.push_back(s[s.size() - 2]);
//	for (int i = 2; i < s.size(); i++) {
//		for (int j = u.size(); j >= 2 && ccw(u[j - 2], u[j-1], s[i]) != CLOCKWISE; j--) {
//			u.pop_back();
//		}
//		u.push_back(s[i]);
//	}
//	for (int i = s.size()-3; i >=0; i--) {
//		for (int j = l.size(); j >= 2 && ccw(l[j - 2], l[j-1], s[i]) != CLOCKWISE; j--) {
//			l.pop_back();
//		}
//		l.push_back(s[i]);
//	}
//	reverse(l.begin(),l.end());
//
//	for (int i = u.size()-2; i >=1; i--) l.push_back(u[i]);
//	return l;
//}

Polygon andrewScan(Polygon s) {
	Polygon u, l;
	if (s.size() < 3) return s;
	sort(s.begin(), s.end());

	u.push_back(s[0]);
	u.push_back(s[1]);

	l.push_back(s[s.size() - 1]);
	l.push_back(s[s.size() - 2]);

	for (int i = 2; i < s.size(); i++) {
		for (int n = u.size(); n >= 2 && (ccw(u[n - 2], u[n - 1], s[i]) == COUNTER_CLOCKWISE); n--) {
			u.pop_back();
		}
		u.push_back(s[i]);
	}
	//for (int i = 0; i < u.size(); i++) {
	//	//u[i].show();
	//	cout << endl;
	//}
	//cout << "AA" << endl;
	for (int i = s.size() - 3; i >= 0; i--) {
		for (int n = l.size(); n >= 2 && ccw(l[n - 2], l[n - 1], s[i]) == COUNTER_CLOCKWISE; n--){
			l.pop_back();
		}
		l.push_back(s[i]);
	}
	//for (int i = 0; i < l.size(); i++) {
	//	//l[i].show();
	//	//cout << endl;
	//}
	reverse(l.begin(), l.end());
	for (int i = u.size() - 2; i >= 1; i--) l.push_back(u[i]);
	//reverse(l.begin(), l.end());

	return l;
}

int main() {
	int n, x, y;
	cin >> n;
	Polygon ans,p;
	for (int i = 0; i < n; i++) {
		cin >> x >> y;
		p.push_back(Point(x, y));
	}
	ans = andrewScan(p);
	cout << ans.size() << endl;
	int head = 0; int minx = 10001; int miny = 10001;
	for (int i = 0; i < ans.size(); i++) {
		if (ans[i].y <= miny) {
			miny = ans[i].y;
			minx = ans[i].x;
			head = i;
		}
	}
	for (int i = 0; i < ans.size(); i++) {
		if (ans[i].y == miny) {
			if (ans[i].x < minx) {
				head = i;
				minx = ans[i].x;
			}
			
		}
	}
	for (int i = 0; i < ans.size(); i++) {
		cout << (int)ans[head].x << " " << (int)ans[head].y << endl;
		head++;
		head %= ans.size();
	}

}
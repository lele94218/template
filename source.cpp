POWERED BY TERRYX

#读写外挂
int Scan() {
    int res = 0, ch, flag = 0;
    if ((ch = getchar()) == '-')
        flag = 1;
    else if (ch >= '0' && ch <= '9')
        res = ch - '0';
    while ((ch = getchar()) >= '0' && ch <= '9')
        res = res * 10 + ch - '0';
    return flag ? -res : res;
}
void Out(int a) {
    if (a > 9)
        Out(a / 10);
    putchar(a % 10 + '0');
}

#########数学&数论#########
#整型二分答案
int L = 0, R = Max;
while (L < R) {
    int M = //(R + L) / 2; (R + L + 1) / 2;
    if (check(M)) //L = M + 1; //L = M;
    else //R = M - 1;
}
cout << L << endl;
#浮点型二分答案
double L = 0, R = Max;
while (R - L > 1e-5) {
    double M = (L + R) / 2;
    if (check(M)) L = M;
    else R = M;
}
printf("%.4lf\n", L);
#所有组合数
ll C[maxn+1][maxn+1];
C[0][0] = 1;
for (int i = 1; i <= maxn; ++ i) {
    for (int j = 0; j <= i; ++ j) {
        C[i][j] = C[i-1][j] + ((j)?(C[i-1][j-1]):0);
    }
}
#二进制i位为1
ll bit(ll mask, ll i) {
    return (mask>>i)&1;
}
#二进制有多少个1
int bit_count(ll x) {
    int cnt = 0;
    while (x) {
        cnt ++;
        x &= (x-1);
    }
    return cnt;
}
#判断n!的因子数(以2为例)
int Count(int n) {
    int ret = 0;
    while (n) {
        ret += (n/2);
        n /= 2;
    }
    return ret;
}
#高斯消元
void gauss_elimination(Matrix A, int n) {
    int i, j, k, r;
    for (i = 0; i < n; ++ i) {
        r = i;
        for (j = i+1; j < n; ++ j)
            if (fabs(A[j][i]) > fabs(A[r][i])) r = j;
        //交换行
        if (r != i) {
            for (j = 0; j <= n; ++ j) swap(A[r][j], A[i][j]);
        }

        //i+1 ~ n 消元
        for (j = n; j >= i; -- j) {
            for (k = i + 1; k < n; ++ k)
                A[k][j] -= A[k][i]/A[i][i] * A[i][j];
        }
    }
    //回代过程
    for (i = n-1; i >= 0; -- i) {
        for (j = i + 1; j < n; ++ j)
            A[i][n] -= A[j][n] * A[i][j];
        A[i][n] /= A[j][i];
    }
}
#快速幂
ll quickpow(ll m, ll n, ll k) {
    ll ans = 1;
    while(n) {
        if(n&1)
            ans = (ans * m) % k;
        n = n >> 1;
        m = (m * m) % k;
    }
    return ans;
}
#快筛素数
ok[0] = ok[1] = 1;
for (int i = 2; i*i < MAXN; ++ i) if (!ok[i])
{
    for (int j = i+i; j < MAXN; j+=i) {
        if (!ok[j]) ok[j] = true;
    }
}
#线性快筛素数
bool ok[maxn];
vector<int> prime;
memset(ok,0,sizeof(ok));
for (int i=2; i<=maxn; i++) {
    if (!ok[i]) {
        prime.push_back(i);
    }
    for (int j=0; j < prime.size() && prime[j]*i<=maxn ; j++) {
        ok[prime[j]*i] = 1;
        if (i%prime[j]==0) break;
    }
}
#求最大公约数
int gcd(int a, int b) {
    return b==0 ? a : gcd(b, a%b);
}
#########图论#########
#邻接表
int p = 0;
struct node {
    int to, w, next;
};
node edge[maxn * maxn];
int head[maxn];
void add_edge(int u, int v, int w) {
    edge[p].to = v;
    edge[p].w = w;
    edge[p].next = head[u];
    head[u] = p;
    ++ p;
}
int main() {
    memset(head, -1, sizeof(head));
    int p = 0;
    for (int i = 0; i < n; ++ i) {
        int u, v, w;
        cin >> u >> v >> w;
        add_edge(u, v, w);
    }
    for (int i = 0; i < n; ++ i) {
        for (int k = head[i]; k != -1; k = edge[k].next) {
            cout << i << " " << edge[k].to << " " << edge[k].w << endl;
        }
    }
}
#强连通分量
struct node {
    int to, w, next;
};
node edge[maxn << 2];
int head[maxn];
int pre[maxn], lowlink[maxn], sccno[maxn], dfs_clock, scc_cnt;
int Stack[maxn], s_m;
ll res, num, cnt;
void dfs(int u) {
    pre[u] = lowlink[u] = ++dfs_clock;
    Stack[++s_m] = u;
    for (int k = head[u]; k != -1; k = edge[k].next) {
        int v = edge[k].to;
        if (!pre[v]) {
            dfs(v);
            lowlink[u] = min(lowlink[u], lowlink[v]);
        }
        else if (!sccno[v]) {
            lowlink[u] = min(lowlink[u], pre[v]);
        }
    }
    if (lowlink[u] == pre[u]) {
        scc_cnt++;
        int Min = INF;
        Set.clear();
        for (;;) {
            //栈中含强连通分量
            int x = Stack[s_m];
            s_m --;
            Min = min(Min, a[x]);
            sccno[x] = scc_cnt;
            if (x == u) break;
        }
    }
}
void find_scc(int n) {
    dfs_clock = scc_cnt = 0;
    memset(sccno, 0, sizeof(sccno));
    memset(pre, 0, sizeof(pre));
    for (int i = 0; i < n; ++ i) {
        if (!pre[i]) dfs(i);
    }
}
#Dijkstra
memset(v, 0, sizeof(v));
for (int i = 1; i <= n; ++ i) d[i] = (i == x) ? 0 : INF; // x 为起点
for (int i = 1; i <= n; ++ i)
{
    int x, m = INF;
    for (int y = 1; y <= n; ++ y) if (!v[y] && d[y] <= m) m = d[x = y];
    v[x] = 1;
    for (int y = 1; y <= n; ++ y) d[y] = min(d[y], d[x] + w[x][y]);
}
for (int i = 0; i <= n; ++ i) for (int j = 0; j <= n; ++ j) w[i][j] = INF;
#最大流算法 - Edmond - Karp
int Map[maxn][maxn], m, n, p[maxn];
bool bfs(int st, int ed) {
    queue <int> q;
    int vis[maxn];
    memset(vis, 0, sizeof(vis));
    memset(p, -1, sizeof(p));
    q.push(st);
    vis[st] = 1;
    while (!q.empty()) {
        int e = q.front();
        if (e == ed) return true;

        q.pop();
        for (int i = 1; i <= n; ++ i) if (Map[e][i] && !vis[i]) {
            vis[i] = 1;
            p[i] = e;
            q.push(i);
        }
    }
    return false;
}
int max_flow (int st, int ed) {
    int u, ans = 0, mn;
    while (bfs(st, ed)) {
        mn = INF;
        u = ed;
        while (p[u] != -1) {
            mn = min(mn, Map[p[u]][u]);
            u = p[u];
        }
        ans += mn;
        u = ed;
        while (p[u] != -1) {
            Map[p[u]][u] -= mn;
            Map[u][p[u]] += mn;
            u = p[u];
        }
    }
    return ans;
}
#二分匹配
int cy[maxn], visy[maxn], nx, ny; //nx 为x个数 ny 为y个数 cy为匹配
bool dfs(int x) {
    for (int k = head[x]; k != -1; k = edge[k].next) {
        int y = edge[k].to;
        if (!visy[y]) {
            visy[y] = 1;
            if (cy[y] < 0 || dfs(cy[y])) {
                cy[y] = x;
                return true;
            }
        }
    }
    return false;
}
int max_match() {
    int res = 0;
    memset(cy, 0xff, sizeof(cy));
    for (int i = 0; i < nx; ++ i) {
        memset(visy, 0, sizeof(visy));
        if (dfs(i)) ++ res;
    }
    return res;
}
#二分图判断 - 建双向边
int col[maxn * maxn];
bool judge(int u) {
    int v;
    for (int k = head[u]; k != -1; k = edge[k].next) {
        v = edge[k].to;
        if (col[v] != -1 && col[v] == col[u]) return false;
        if (col[v] == -1) {
            col[v] = !col[u];
            if (!judge(v)) return false;
        }
    }
    return true;
}
#最大二分匹配 - 匈牙利 - 邻接表
int uN, vN;//u,v数目
int linker[maxn];
bool used[maxn];
bool dfs(int u) { //从左边开始找增广路径
    int v;
    for (int k = head[u]; k != -1; k = edge[k].next) {
        int v = edge[k].to;
        if(!used[v]) {
            used[v] = true;
            if(linker[v] == -1 || dfs(linker[v])) {
                //找增广路，反向
                linker[v] = u;
                return true;
            }
        }
    }
    return false;
}
int max_match() {
    int res=0;
    int u;
    memset(linker,-1,sizeof(linker));
    for(u = 0; u < uN; u++) {
        memset(used, 0, sizeof(used));
        if(dfs(u)) res ++;
    }
    return res;
}
#最大二分匹配 - 匈牙利 - 邻接矩阵
int uN, vN;//u,v数目
int edge[maxn][maxn];
int linker[maxn];
bool used[maxn];
bool dfs(int u) { //从左边开始找增广路径
    int v;
    for(v = 0; v < vN; ++ v) {
        if(edge[u][v] && !used[v]) {
            used[v] = true;
            if(linker[v] == -1 || dfs(linker[v])) {
                //找增广路，反向
                linker[v] = u;
                return true;
            }
        }
    }
    return false;
}
int max_match() {
    int res=0;
    int u;
    memset(linker,-1,sizeof(linker));
    for(u = 0; u < uN; u++) {
        memset(used, 0, sizeof(used));
        if(dfs(u)) res ++;
    }
    return res;
}
#最大二分匹配 - Hopcroft-Carp - 邻接表
int Mx[maxn],My[maxn],uN,vN;
int dx[maxn],dy[maxn],dis;
bool vis[maxn];
bool search_path() {
    queue<int> Q;
    dis = INF;
    memset(dx, -1, sizeof(dx));
    memset(dy, -1, sizeof(dy));
    for (int i = 0; i < uN; i++)
        if(Mx[i] == -1) {
            Q.push(i);
            dx[i] = 0;
        }
    while (!Q.empty()) {
        int u = Q.front();
        Q.pop();
        if (dx[u] > dis)  break;
        int v;
        for (int k = head[u]; k != -1; k = edge[k].next) {
            v = edge[k].to;
            if (dy[v] == -1) {
                dy[v] = dx[u] + 1;
                if (My[v] == -1)  dis = dy[v];
                else {
                    dx[My[v]] = dy[v]+1;
                    Q.push(My[v]);
                }
            }
        }
    }
    return dis != INF;
}
bool dfs(int u) {
    int v;
    for (int k = head[u]; k != -1; k = edge[k].next) {
        v = edge[k].to;
        if (!vis[v] && dy[v]==dx[u]+1) {
            vis[v] = 1;
            if (My[v] != -1 && dy[v] == dis) continue;
            if(My[v] == -1 || dfs(My[v])) {
                My[v] = u;
                Mx[u] = v;
                return true;
            }
        }
    }
    return false;
}
int max_match() {
    int res = 0;
    memset(Mx, -1, sizeof(Mx));
    memset(My, -1, sizeof(My));
    while (search_path()) {
        memset(vis, 0, sizeof(vis));
        for (int i = 0; i < uN; i ++)
            if (Mx[i] == -1 && dfs(i))  res ++;
    }
    return res;
}
#最大二分匹配 - Hopcroft-Carp - 邻接矩阵
int edge[maxn][maxn],Mx[maxn],My[maxn],uN,vN;
int dx[maxn],dy[maxn],dis;
bool vis[maxn];
bool search_path() {
    queue<int> Q;
    dis = INF;
    memset(dx, -1, sizeof(dx));
    memset(dy, -1, sizeof(dy));
    for (int i = 0; i < uN; i++)
        if(Mx[i] == -1) {
            Q.push(i);
            dx[i] = 0;
        }
    while (!Q.empty()) {
        int u = Q.front();
        Q.pop();
        if (dx[u] > dis)  break;
        for (int v = 0; v < vN; v ++) {
            if (edge[u][v]&&dy[v] == -1) {
                dy[v] = dx[u] + 1;
                if (My[v] == -1)  dis = dy[v];
                else {
                    dx[My[v]] = dy[v]+1;
                    Q.push(My[v]);
                }
            }
        }
    }
    return dis != INF;
}
bool dfs(int u) {
    for (int v=0; v<vN; v++) {
        if (!vis[v] && edge[u][v] && dy[v]==dx[u]+1) {
            vis[v] = 1;
            if (My[v] != -1 && dy[v] == dis) continue;
            if(My[v] == -1 || dfs(My[v])) {
                My[v] = u;
                Mx[u] = v;
                return true;
            }
        }
    }
    return false;
}
int max_match() {
    int res = 0;
    memset(Mx, -1, sizeof(Mx));
    memset(My, -1, sizeof(My));
    while (search_path()) {
        memset(vst, 0, sizeof(vst));
        for (int i = 0; i < uN; i ++)
            if (Mx[i] == -1 && dfs(i))  res ++;
    }
    return res;
}
#########数据结构#########
#归并排序求逆序数
void merge_sort(P* A, int x, int y, P *T) {
    if (y - x > 1) {
        int m = (y + x) >> 1;
        int p = x, q = m, i = x;
        merge_sort(A, x, m, T);
        merge_sort(A, m, y, T);
        while (p < m || q < y) {
            if(q >= y || (p < m && A[p].b <= A[q].b)) { //最右边为排序条件
                T[i++] = A[p++];
            } else {
                T[i++] = A[q++];
                ans += ll(m-p);
            }
        }
        for (i = x; i < y; i ++)
            A[i] = T[i];
    }
}
#RMQ 用于维护[l,r]的最小值（数值，下标） 注意：数组一定要从1开始，不然会Runtime Error
#数值
void RMQ_init(int n) {
    for (int i = 1; i <= n; ++ i) f[i][0] = a[i];
    for (int j = 1; j <= int( log(double(n))/log(2.0) ); ++ j)
        for (int i = 1; i <= n-(1<<(j-1)); ++ i)
            f[i][j] = min(f[i][j-1],f[i+(1<<(j-1))][j-1]);
}
int RMQ(int L, int R) {
    int t = int( log(double(R-L+1))/log(2.0) );
    return min(f[L][t],f[R+1-(1<<t)][t]);
}
#下标
void RMQ_init(int n) {
    for (int i = 0; i < n; ++ i) m[i][0] = i;
    for (int i = 0; i < n; ++ i)
        for (int j = 1; i + (1<<j) - 1 < n; ++ j)
            m[i][j] = (a[m[i][j-1]] < a[m[i+(1<<(j-1))][j-1]]) ? m[i][j-1] : m[i+(1<<(j-1))][j-1];
}
int RMQ(int l, int r) {
    int k = 0;
    while ((1<<(k+1)) <= r-l+1) ++ k;
    if (a[m[l][k]] < a[m[r-(1<<k)+1][k]]) return m[l][k];
    else return m[r-(1<<k)+1][k];
}
#树状数组(点操作前缀和)
int num[maxn << 1], maxn;
int lowbit(int x) {return x&(-x);}
void update(int x, int v) {
    for (; x <= maxn; x += lowbit(x)) {
        num[x] += v;
    }
}
ll query(int x) {
    ll ret = 0;
    for (; x > 0; x -= lowbit(x)) {
        ret += ll(num[x]);
    }
    return ret;
}
#线段树_lazy标记(区间操作)
#define Lson l, mid, rt << 1
#define Rson mid + 1, r, rt << 1 | 1
using namespace std;
typedef long long ll;
ll sum[maxn << 2];
ll now[maxn << 2];
bool mark[maxn << 2];
void push_down(int rt) {
    if (mark[rt]) {
        mark[rt << 1] = 1;
        mark[rt << 1 | 1] = 1;
        //...code...//
        mark[rt] = 0;
    }
}
void push_up(int rt) {
    sum[rt] = sum[rt << 1] + sum[rt << 1 | 1];
}
void build(int l, int r, int rt) {
    if (l == r) {
        //...初始化...//
        return;
    }
    int mid = (l + r) >> 1;
    build(Lson);
    build(Rson);
    push_up(rt);
    push_up1(rt);
}
//单点更新
void update(int p, ll add, int l, int r, int rt) {
    if (l == r) {
        //...code...//
        return;
    }
    push_down(rt);
    int mid = (l + r) >> 1;
    if (p <= mid) {
        update(p, add, Lson);
    } else {
        update(p, add, Rson);
    }
    push_up(rt);
}
//区间更新
void update1(int L, int R, int l, int r, int rt) {
    if (L <= l && r <= R) {
        //...code...//
        return;
    }
    push_down(rt);
    int mid = (l + r) >> 1;
    if (L <= mid)
        update1(L, R, Lson);
    if (R > mid)
        update1(L, R, Rson);
    push_up(rt);
}
//查询区间
ll query(int L, int R, int l, int r, int rt) {
    if (L <= l && r <= R) {
        return sum[rt];
    }
    push_down(rt);
    ll ret = 0;
    int mid = (l + r) >> 1;
    if (L <= mid) {
        ret += query(L, R, Lson);
    }
    if (R > mid) {
        ret += query(L, R, Rson);
    }
    return ret;
}
#Trie
struct Trie {
    int ch[maxnode][sigma_size];
    int val[maxnode];//单词权值
    int sz;
    void init() {
        memset(ch[0], 0, sizeof(ch[0]));
        sz = 1;
    }
    int P(char ch) {
        if ('0' <= ch && ch <= '9') return ch - '0';
        if ('a' <= ch && ch <= 'z') return 10 + ch - 'a';
        if ('A' <= ch && ch <= 'Z') return 36 + ch - 'A';
        return 62;
    }
    void Insert(char *s, int v) {
        int u = 0;
        for (int i = 0; s[i]; ++ i) {
            int c = idx(s[i]);
            if (!ch[u][c]) {
                memset(ch[sz], 0, sizeof(ch[sz]));
                val[sz] = 0;
                ch[u][c] = sz++;
            }
            u = ch[u][c];
        }
        val[u] = v;
    }
    void Query(char *s, int p) { //查找前缀个数
        int cur = 0;

        for (int i = 0; s[i]; ++ i) {
            int c = idx(s[i]);
            if (!ch[cur][c]) break;
            if (val[ch[cur][c]] == -1) {
                //查找到单词
            }
            cur = ch[cur][c];

        }
    }
}tx;

struct t_Trie {
    struct Node {
        int ch[sigma_size], value;
        void init() {
            memset(ch, 0, sizeof(ch));
            value = 0;
        }
    };
    Node node[maxnode];
    int used, root;
    int P(char ch) {
        if ('0' <= ch && ch <= '9') return ch - '0';
        if ('a' <= ch && ch <= 'z') return 10 + ch - 'a';
        if ('A' <= ch && ch <= 'Z') return 36 + ch - 'A';
        return 62;
    }
    int newNode() {
        node[used].init();
        return used ++;
    }
    void init() {
        used = 0;
        root = newNode();
    }
    void Add(char *s, int v) {
        int x = root;
        for (int i = 0; s[i]; ++ i) {
            int p = P(s[i]);
            if (!node[x].ch[p]) node[x].ch[p] = newNode();
            x = node[x].ch[p];
        }
        node[x].value = v;
    }
}tx;

struct List_Trie {
    struct Node {
        int val;//记录结点字母
        int fir, nxt;
        void init() {
            cnt = 0;
            fir = nxt = val = -1;
        }
    };
    Node node[maxnode];
    int used, root;
    int P(char ch) {
        if ('0' <= ch && ch <= '9') return ch - '0';
        if ('a' <= ch && ch <= 'z') return 10 + ch - 'a';
        if ('A' <= ch && ch <= 'Z') return 36 + ch - 'A';
        return 62;
    }
    int newNode() {
        node[used].init();
        return used ++;
    }
    void init() {
        used = 0;
        root = newNode();
    }
    int Find(int x, int p) {
        for (int i = node[x].fir; ~i; i = node[i].nxt) {
            if (node[i].val == p) return i;
        }
        return -1;
    }
    void Add(char *s) {
        int x = root;
        for (int i = 0; s[i]; ++ i) {
            int p = P(s[i]);
            int ix = Find(x, p);
            if (ix == -1) {
                ix = newNode();
                node[ix].val = p;
                node[ix].nxt = node[x].fir;
                node[x].fir = ix;
            }
        }
    }
}tx;

#########计算几何#########
//自己总结的

int dcmp(double x) {
  if(fabs(x) < eps) return 0; else return x < 0 ? -1 : 1;
}

struct Point {
    double x, y;
    Point(double x = 0, double y = 0): x(x), y(y) { }
};
typedef Point Vector;

Point operator + (Point A, Point B) { return Point(A.x+B.x, A.y+B.y); }
Point operator - (Point A, Point B) { return Point(A.x-B.x, A.y-B.y); }
Point operator * (Point A, double p) { return Point(A.x*p, A.y*p); }
Point operator / (Point A, double p) { return Point(A.x/p, A.y/p); }

bool operator < (const Point& a, const Point& b) {
  return a.x < b.x || (a.x == b.x && a.y < b.y);
}
bool operator == (const Point& a, const Point &b) {
  return dcmp(a.x-b.x) == 0 && dcmp(a.y-b.y) == 0;
}

double Cross(Point A, Point B) { return A.x*B.y - A.y*B.x; }

struct Line {
  Point p;    // 直线上任意一点
  Point v;   // 方向向量
  double ang; // 极角，即从x正半轴旋转到向量v所需要的角（弧度）
  Line() {}
  Line(Point P, Point v): p(P),v(v){ ang = atan2(v.y, v.x); }
  Point point(double a) { // 线段两端点为 p 和 point(1)
        return p+(v*a);
  }
  bool operator < (const Line& L) const {
    return ang < L.ang;
  }
  void print() {
    p.print(), cout << " "; point(1).print(); cout << endl;
  }
};
//判断平行线段是否共线, 前提为平行
bool SegmentCollinear(line a, line b) {
    Vector _1, _2;
    _1 = a.point(1) - a.p;
    _2 = b.p - a.p;
    if (Cross(_1, _2) == 0) return true;
    return false;
}





//参考
/***********三分法求函数极值*************/
void solve()
{
    double L, R, m, mm, mv, mmv;
    while (L + eps < R)
    {
        m = (L + R) / 2;
        mm = (m + R) / 2;
        mv = calc(m);
        mmv = calc(mm);
        if (mv <= mmv) R = mm; //三分法求最大值时改为mv>=mmv
        else L = m;
    }
}
/*************基础***********/
int dcmp(double x) {
  if(fabs(x) < eps) return 0; else return x < 0 ? -1 : 1;
}
struct Point {
  double x, y;
  Point(double x=0, double y=0):x(x),y(y) { }
};
Point operator + (Point A, Point B) { return Point(A.x+B.x, A.y+B.y); }
Point operator - (Point A, Point B) { return Point(A.x-B.x, A.y-B.y); }
Point operator * (Point A, double p) { return Point(A.x*p, A.y*p); }
Point operator / (Point A, double p) { return Point(A.x/p, A.y/p); }

bool operator < (const Point& a, const Point& b) {
  return a.x < b.x || (a.x == b.x && a.y < b.y);
}

bool operator == (const Point& a, const Point &b) {
  return dcmp(a.x-b.x) == 0 && dcmp(a.y-b.y) == 0;
}

double Dot(Point A, Point B) { return A.x*B.x + A.y*B.y; }
double Length(Point A) { return sqrt(Dot(A, A)); }
double Angle(Point A, Point B) { return acos(Dot(A, B) / Length(A) / Length(B)); }
double angle(Point v) { return atan2(v.y, v.x); }
double Cross(Point A, Point B) { return A.x*B.y - A.y*B.x; }
/*
向量叉积
若 P × Q > 0 , 则P在Q的顺时针方向。
若 P × Q < 0 , 则P在Q的逆时针方向。
若 P × Q = 0 , 则P与Q共线，但可能同向也可能反向。
*/
Point vecunit(Point x){ return x / Length(x);} //单位向量
Point Normal(Point x) { return Point(-x.y, x.x) / Length(x);} //垂直法向量
Point Rotate(Point A, double rad) {
  return Point(A.x*cos(rad)-A.y*sin(rad), A.x*sin(rad)+A.y*cos(rad));
}
double Area2(const Point A, const Point B, const Point C) { return Length(Cross(B-A, C-A)); }

/****************直线与线段**************/

//求直线p+tv和q+tw的交点 Cross(v, w) == 0无交点
Point GetLineIntersection(Point p, Point v, Point q, Point w)
{
    Point u = p-q;
    double t = Cross(w, u) / Cross(v, w);
    return p + v*t;
}

//点p在直线ab的投影
Point GetLineProjection(Point P, Point A, Point B) {
  Point v = B-A;
  return A+v*(Dot(v, P-A) / Dot(v, v));
}

//点到直线距离
double DistanceToLine(Point P, Point A, Point B) {
  Point v1 = B - A, v2 = P - A;
  return fabs(Cross(v1, v2)) / Length(v1); // 如果不取绝对值，得到的是有向距离
}
//点在p线段上
bool OnSegment(Point p, Point a1, Point a2) {
  return dcmp(Cross(a1-p, a2-p)) == 0 && dcmp(Dot(a1-p, a2-p)) < 0; //线段包含端点时改成<=
}
// 过两点p1, p2的直线一般方程ax+by+c=0
// (x2-x1)(y-y1) = (y2-y1)(x-x1)
void getLineGeneralEquation(const Point& p1, const Point& p2, double& a, double& b, double &c) {
  a = p2.y-p1.y;
  b = p1.x-p2.x;
  c = -a*p1.x - b*p1.y;
}
//点到线段距离
double DistanceToSegment(Point p, Point a, Point b)
{
    if(a == b) return Length(p-a);
    Point v1 = b-a, v2 = p-a, v3 = p-b;
    if(dcmp(Dot(v1, v2)) < 0) return Length(v2);
    else if(dcmp(Dot(v1, v3)) > 0) return Length(v3);
    else return fabs(Cross(v1, v2)) / Length(v1);
}
//两线段最近距离
double dis_pair_seg(Point p1, Point p2, Point p3, Point p4)
{
    return min(min(DistanceToSegment(p1, p3, p4), DistanceToSegment(p2, p3, p4)),
     min(DistanceToSegment(p3, p1, p2), DistanceToSegment(p4, p1, p2)));
}
//线段相交判定
bool SegmentItersection(Point a1, Point a2, Point b1, Point b2)
{
    double c1 = Cross(a2-a1, b1-a1), c2 = Cross(a2-a1, b2-a1),
    c3 = Cross(b2-b1, a1-b1), c4 = Cross(b2-b1, a2-b1);
    return dcmp(c1)*dcmp(c2) < 0 && dcmp(c3)*dcmp(c4) < 0;
}
// 有向直线。它的左边就是对应的半平面
struct Line {
  Point p;    // 直线上任意一点
  Point v;   // 方向向量
  double ang; // 极角，即从x正半轴旋转到向量v所需要的角（弧度）
  Line() {}
  Line(Point P, Point v): p(P),v(v){ ang = atan2(v.y, v.x); }
  Point point(double a) {
        return p+(v*a);
  }
  bool operator < (const Line& L) const {
    return ang < L.ang;
  }
}
};
//两直线交点
Point GetLineIntersection(Line a, Line b) {
  return GetLineIntersection(a.p, a.v, b.p, b.v);
}

// 点p在有向直线L的左边（线上不算）
bool OnLeft(const Line& L, const Point& p) {
  return Cross(L.v, p-L.P) >= 0;
}

// 二直线交点，假定交点惟一存在
Point GetLineIntersection(const Line& a, const Line& b) {
  Point u = a.P-b.P;
  double t = Cross(b.v, u) / Cross(a.v, b.v);
  return a.P+a.v*t;
}

// 半平面交主过程
vector<Point> HalfplaneIntersection(vector<Line> L) {
  int n = L.size();
  sort(L.begin(), L.end()); // 按极角排序

  int first, last;         // 双端队列的第一个元素和最后一个元素的下标
  vector<Point> p(n);      // p[i]为q[i]和q[i+1]的交点
  vector<Line> q(n);       // 双端队列
  vector<Point> ans;       // 结果

  q[first=last=0] = L[0];  // 双端队列初始化为只有一个半平面L[0]
  for(int i = 1; i < n; i++) {
    while(first < last && !OnLeft(L[i], p[last-1])) last--;
    while(first < last && !OnLeft(L[i], p[first])) first++;
    q[++last] = L[i];
    if(fabs(Cross(q[last].v, q[last-1].v)) < eps) { // 两向量平行且同向，取内侧的一个
      last--;
      if(OnLeft(q[last], L[i].P)) q[last] = L[i];
    }
    if(first < last) p[last-1] = GetLineIntersection(q[last-1], q[last]);
  }
  while(first < last && !OnLeft(q[first], p[last-1])) last--; // 删除无用平面
  if(last - first <= 1) return ans; // 空集
  p[last] = GetLineIntersection(q[last], q[first]); // 计算首尾两个半平面的交点

  // 从deque复制到输出中
  for(int i = first; i <= last; i++) ans.push_back(p[i]);
  return ans;
}

/***********多边形**************/
//多边形有向面积
double PolygonArea(vector<Point> p) {
  int n = p.size();
  double area = 0;
  for(int i = 1; i < n-1; i++)
    area += Cross(p[i]-p[0], p[i+1]-p[0]);
  return area/2;
}

//多边形重心 点集逆时针给出
Point PolyGravity(Point *p, int n) {
    Point tmp, g = Point(0, 0);
    double sumArea = 0, area;
    for (int i=2; i<n; ++i) {
        area = Cross(p[i-1]-p[0], p[i]-p[0]);
        sumArea += area;
        tmp.x = p[0].x + p[i-1].x + p[i].x;
        tmp.y = p[0].y + p[i-1].y + p[i].y;
        g.x += tmp.x * area;
        g.y += tmp.y * area;
    }
    g.x /= (sumArea * 3.0); g.y /= (sumArea * 3.0);
    return g;
}

// 点集凸包
// 如果不希望在凸包的边上有输入点，把两个 <= 改成 <
// 注意：输入点集会被修改
vector<Point> ConvexHull(vector<Point>& p) {
  // 预处理，删除重复点
  sort(p.begin(), p.end());
  p.erase(unique(p.begin(), p.end()), p.end());

  int n = p.size();
  int m = 0;
  vector<Point> ch(n+1);
  for(int i = 0; i < n; i++) {
    while(m > 1 && Cross(ch[m-1]-ch[m-2], p[i]-ch[m-2]) <= 0) m--;
    ch[m++] = p[i];
  }
  int k = m;
  for(int i = n-2; i >= 0; i--) {
    while(m > k && Cross(ch[m-1]-ch[m-2], p[i]-ch[m-2]) <= 0) m--;
    ch[m++] = p[i];
  }
  if(n > 1) m--;
  ch.resize(m);
  return ch;
}
//判断点是否在多边形内
int isPointInPolygon(Point p, Polygon poly)
{
    int wn = 0;
    int n = poly.size();
    for (int i = 0; i < n; i++)
    {
        if (OnSegment(p, poly[i], poly[(i + 1) % n])) return -1;    //边界
        int k = dcmp(Cross(poly[(i + 1) % n] - poly[i], p - poly[i]));
        int d1 = dcmp(poly[i].y - p.y);
        int d2 = dcmp(poly[(i + 1) % n].y - p.y);
        if (k > 0 && d1 <= 0 && d2 > 0) wn++;
        if (k < 0 && d2 <= 0 && d1 > 0) wn--;
    }
    if (wn != 0) return 1;  //内部
    return 0;   //外部
}
// 凸包直径返回 点集直径的平方
int diameter2(vector<Point>& points) {
  vector<Point> p = ConvexHull(points);
  int n = p.size();
  if(n == 1) return 0;
  if(n == 2) return Dist2(p[0], p[1]);
  p.push_back(p[0]); // 免得取模
  int ans = 0;
  for(int u = 0, v = 1; u < n; u++) {
    // 一条直线贴住边p[u]-p[u+1]
    for(;;) {
      int diff = Cross(p[u+1]-p[u], p[v+1]-p[v]);
      if(diff <= 0) {
        ans = max(ans, Dist2(p[u], p[v])); // u和v是对踵点
        if(diff == 0) ans = max(ans, Dist2(p[u], p[v+1])); // diff == 0时u和v+1也是对踵点
        break;
      }
      v = (v + 1) % n;
    }
  }
  return ans;
}
//两凸包最近距离
double RC_Distance(Point *ch1, Point *ch2, int n, int m)
{
    int q=0, p=0;
    REP(i, n) if(ch1[i].y-ch1[p].y < -eps) p=i;
    REP(i, m) if(ch2[i].y-ch2[q].y > eps) q=i;
    ch1[n]=ch1[0];  ch2[m]=ch2[0];

    double tmp, ans=1e100;
    REP(i, n)
    {
        while((tmp = Cross(ch1[p+1]-ch1[p], ch2[q+1]-ch1[p]) - Cross(ch1[p+1]-ch1[p], ch2[q]- ch1[p])) > eps)
            q=(q+1)%m;
        if(tmp < -eps) ans = min(ans,DistanceToSegment(ch2[q],ch1[p],ch1[p+1]));
        else ans = min(ans,dis_pair_seg(ch1[p],ch1[p+1],ch2[q],ch2[q+1]));
        p=(p+1)%n;
    }
    return ans;
}
//凸包最大内接三角形
double RC_Triangle(Point* res,int n)// 凸包最大内接三角形
{
     if(n<3)    return 0;
     double ans=0, tmp;
     res[n] = res[0];
     int j, k;
     REP(i, n)
     {
         j = (i+1)%n;
         k = (j+1)%n;
         while((j != k) && (k != i))
         {
              while(Cross(res[j] - res[i], res[k+1] - res[i]) > Cross(res[j] - res[i], res[k] - res[i])) k= (k+1)%n;
              tmp = Cross(res[j] - res[i], res[k] - res[i]);if(tmp > ans) ans = tmp;
              j = (j+1)%n;
         }
     }
     return ans;
}
//模拟退火求费马点 保存在ptres中
double fermat_point(Point *pt, int n, Point& ptres)
{
    Point u, v;
    double step = 0.0, curlen, explen, minlen;
    int i, j, k, idx;
    bool flag;
    u.x = u.y = v.x = v.y = 0.0;
    REP(i, n)
    {
        step += fabs(pt[i].x) + fabs(pt[i].y);
        u.x += pt[i].x;
        u.y += pt[i].y;
    }
    u.x /= n;
    u.y /= n;
    flag = 0;
    while(step > eps)
    {
        for(k = 0; k < 10; step /= 2, ++k)
            for(i = -1; i <= 1; ++i)
                for(j = -1; j <= 1; ++j)
                {
                    v.x = u.x + step*i;
                    v.y = u.y + step*j;
                    curlen = explen = 0.0;
                        REP(idx, n)
                    {
                        curlen += dist(u, pt[idx]);
                        explen += dist(v, pt[idx]);
                    }
                    if(curlen > explen)
                    {
                        u = v;
                        minlen = explen;
                        flag = 1;
                    }
                }
    }
    ptres = u;
    return flag ? minlen : curlen;
}
//最近点对
bool cmpxy(const Point& a, const Point& b)
{
    if(a.x != b.x)
        return a.x < b.x;
    return a.y < b.y;
}
bool cmpy(const int& a, const int& b)
{
    return point[a].y < point[b].y;
}
double Closest_Pair(int left, int right)
{
    double d = INF;
    if(left==right)
        return d;
    if(left + 1 == right)
        return dis(left, right);
    int mid = (left+right)>>1;
    double d1 = Closest_Pair(left,mid);
    double d2 = Closest_Pair(mid+1,right);
    d = min(d1,d2);
    int i,j,k=0;
    //分离出宽度为d的区间
    for(i = left; i <= right; i++)
    {
        if(fabs(point[mid].x-point[i].x) <= d)
            tmpt[k++] = i;
    }
    sort(tmpt,tmpt+k,cmpy);
    //线性扫描
    for(i = 0; i < k; i++)
    {
        for(j = i+1; j < k && point[tmpt[j]].y-point[tmpt[i]].y<d; j++)
        {
            double d3 = dis(tmpt[i],tmpt[j]);
            if(d > d3)
                d = d3;
        }
    }
    return d;
}

/************圆************/
struct Circle
{
    Point c;
    double r;
    Circle(){}
    Circle(Point c, double r):c(c), r(r){}
    Point point(double a) //根据圆心角求点坐标
    {
        return Point(c.x+cos(a)*r, c.y+sin(a)*r);
    }
};
//求a点到b点(逆时针)在的圆上的圆弧长度
double D(Point a,Point b,int id)
{
    double ang1,ang2;
    Point v1,v2;
    v1=a-Point(C[id].c.x,C[id].c.y);
    v2=b-Point(C[id].c.x,C[id].c.y);
    ang1=atan2(v1.y,v1.x);
    ang2=atan2(v2.y,v2.x);
    if(ang2<ang1) ang2+=2*pi;
    return C[id].r*(ang2-ang1);
}
//直线与圆交点 返回个数
int getLineCircleIntersection(Line L, Circle C, double& t1, double& t2, vector<Point>& sol){
  double a = L.v.x, b = L.p.x - C.c.x, c = L.v.y, d = L.p.y - C.c.y;
  double e = a*a + c*c, f = 2*(a*b + c*d), g = b*b + d*d - C.r*C.r;
  double delta = f*f - 4*e*g; // 判别式
  if(dcmp(delta) < 0) return 0; // 相离
  if(dcmp(delta) == 0) { // 相切
    t1 = t2 = -f / (2 * e); sol.push_back(L.point(t1));
    return 1;
  }
  // 相交
  t1 = (-f - sqrt(delta)) / (2 * e); sol.push_back(L.point(t1));
  t2 = (-f + sqrt(delta)) / (2 * e); sol.push_back(L.point(t2));
  return 2;
}
//两圆交点 返回个数
int getCircleCircleIntersection(Circle C1, Circle C2, vector<Point>& sol) {
  double d = Length(C1.c - C2.c);
  if(dcmp(d) == 0) {
    if(dcmp(C1.r - C2.r) == 0) return -1; // 重合，无穷多交点
    return 0;
  }
  if(dcmp(C1.r + C2.r - d) < 0) return 0;
  if(dcmp(fabs(C1.r-C2.r) - d) > 0) return 0;

  double a = angle(C2.c - C1.c);
  double da = acos((C1.r*C1.r + d*d - C2.r*C2.r) / (2*C1.r*d));
  Point p1 = C1.point(a-da), p2 = C1.point(a+da);

  sol.push_back(p1);
  if(p1 == p2) return 1;
  sol.push_back(p2);
  return 2;
}
//P到圆的切线
//v[i]是第i条切线的向量, 返回切线数
int getTangents(Point p, Circle C, Point* v)
{
    Point u = C.c - p;
    double dist = Length(u);
    if (dist < C.r) return 0;
    else if (dcmp(dist - C.r) == 0)
    {
        //P在圆上,只有一条切线
        v[0] = Rotate(u, PI / 2);
        return 1;
    }
    else
    {
        double ang = asin(C.r / dist);
        v[0] = Rotate(u, -ang);
        v[1] = Rotate(u, +ang);
        return 2;
    }
}

//两圆的公切线, -1表示无穷条切线
int getTangents(Circle A, Circle B, Point* a, Point* b)
{
    int cnt = 0;
    if (A.r < B.r) swap(A, B), swap(a, b);
    ///****************************
    int d2 = (A.c.x - B.c.x) * (A.c.x - B.c.x) + (A.c.y - B.c.y) * (A.c.y - B.c.y);
    int rdiff = A.r - B.r;
    int rsum = A.r + B.r;
    if (d2 < rdiff * rdiff) return 0;   //内含

    ///***************************************
    double base = atan2(B.c.y - A.c.y, B.c.x - A.c.x);
    if (d2 == 0 && A.r == B.r) return -1;    //无线多条切线
    if (d2 == rdiff * rdiff)    //内切, 1条切线
    {
        ///**********************
        a[cnt] = A.point(base); b[cnt] = B.point(base); cnt++;
        return 1;
    }
    //有外公切线
    double ang = acos((A.r - B.r) / sqrt(d2));
    a[cnt] = A.point(base + ang); b[cnt] = B.point(base + ang); cnt++;
    a[cnt] = A.point(base - ang); b[cnt] = B.point(base - ang); cnt++;
    if (d2 == rsum * rsum)  //一条内公切线
    {
        a[cnt] = A.point(base); b[cnt] = B.point(PI + base); cnt++;
    }
    else if (d2 > rsum * rsum)  //两条内公切线
    {
        double ang = acos((A.r + B.r) / sqrt(d2));
        a[cnt] = A.point(base + ang); b[cnt] = B.point(PI + base + ang); cnt++;
        a[cnt] = A.point(base - ang); b[cnt] = B.point(PI + base - ang); cnt++;
    }
    return cnt;
}

//三角形外接圆
Circle CircumscribedCircle(Point p1, Point p2, Point p3) {
  double Bx = p2.x-p1.x, By = p2.y-p1.y;
  double Cx = p3.x-p1.x, Cy = p3.y-p1.y;
  double D = 2*(Bx*Cy-By*Cx);
  double cx = (Cy*(Bx*Bx+By*By) - By*(Cx*Cx+Cy*Cy))/D + p1.x;
  double cy = (Bx*(Cx*Cx+Cy*Cy) - Cx*(Bx*Bx+By*By))/D + p1.y;
  Point p = Point(cx, cy);
  return Circle(p, Length(p1-p));
}

//三角形内切圆
Circle InscribedCircle(Point p1, Point p2, Point p3) {
  double a = Length(p2-p3);
  double b = Length(p3-p1);
  double c = Length(p1-p2);
  Point p = (p1*a+p2*b+p3*c)/(a+b+c);
  return Circle(p, DistanceToLine(p, p1, p2));
}

// 过点p到圆C的切线。v[i]是第i条切线的向量。返回切线条数
int getTangents(Point p, Circle C, Point* v) {
  Point u = C.c - p;
  double dist = Length(u);
  if(dist < C.r) return 0;
  else if(dcmp(dist - C.r) == 0) { // p在圆上，只有一条切线
    v[0] = Rotate(u, PI/2);
    return 1;
  } else {
    double ang = asin(C.r / dist);
    v[0] = Rotate(u, -ang);
    v[1] = Rotate(u, +ang);
    return 2;
  }
}

//所有经过点p 半径为r 且与直线L相切的圆心
vector<Point> CircleThroughPointTangentToLineGivenRadius(Point p, Line L, double r) {
  vector<Point> ans;
  double t1, t2;
  getLineCircleIntersection(L.move(-r), Circle(p, r), t1, t2, ans);
  getLineCircleIntersection(L.move(r), Circle(p, r), t1, t2, ans);
  return ans;
}

//半径为r 与a b两直线相切的圆心
vector<Point> CircleTangentToLinesGivenRadius(Line a, Line b, double r) {
  vector<Point> ans;
  Line L1 = a.move(-r), L2 = a.move(r);
  Line L3 = b.move(-r), L4 = b.move(r);
  ans.push_back(GetLineIntersection(L1, L3));
  ans.push_back(GetLineIntersection(L1, L4));
  ans.push_back(GetLineIntersection(L2, L3));
  ans.push_back(GetLineIntersection(L2, L4));
  return ans;
}

//与两圆相切 半径为r的所有圆心
vector<Point> CircleTangentToTwoDisjointCirclesWithRadius(Circle c1, Circle c2, double r) {
  vector<Point> ans;
  Point v = c2.c - c1.c;
  double dist = Length(v);
  int d = dcmp(dist - c1.r -c2.r - r*2);
  if(d > 0) return ans;
  getCircleCircleIntersection(Circle(c1.c, c1.r+r), Circle(c2.c, c2.r+r), ans);
  return ans;
}

//多边形与圆相交面积
Point GetIntersection(Line a, Line b) //线段交点
{
    Point u = a.p-b.p;
    double t = Cross(b.v, u) / Cross(a.v, b.v);
    return a.p + a.v*t;
}
bool InCircle(Point x, Circle c) { return dcmp(c.r - Length(c.c - x)) >= 0;}
bool OnCircle(Point x, Circle c) { return dcmp(c.r - Length(c.c - x)) == 0;}
//线段与圆的交点
int getSegCircleIntersection(Line L, Circle C, Point* sol)
{
    Point nor = normal(L.v);
    Line pl = Line(C.c, nor);
    Point ip = GetIntersection(pl, L);
    double dis = Length(ip - C.c);
    if (dcmp(dis - C.r) > 0) return 0;
    Point dxy = vecunit(L.v) * sqrt(sqr(C.r) - sqr(dis));
    int ret = 0;
    sol[ret] = ip + dxy;
    if (OnSegment(sol[ret], L.p, L.point(1))) ret++;
    sol[ret] = ip - dxy;
    if (OnSegment(sol[ret], L.p, L.point(1))) ret++;
    return ret;
}
double SegCircleArea(Circle C, Point a, Point b) //线段切割圆
{
    double a1 = angle(a - C.c);
    double a2 = angle(b - C.c);
    double da = fabs(a1 - a2);
    if (da > PI) da = PI * 2.0 - da;
    return dcmp(Cross(b - C.c, a - C.c)) * da * sqr(C.r) / 2.0;
}

double PolyCiclrArea(Circle C, Point *p, int n)//多边形与圆相交面积
{
    double ret = 0.0;
    Point sol[2];
    p[n] = p[0];
    REP(i, n)
    {
        double t1, t2;
        int cnt = getSegCircleIntersection(Line(p[i], p[i+1]-p[i]), C, sol);
        if (cnt == 0)
        {
            if (!InCircle(p[i], C) || !InCircle(p[i+1], C)) ret += SegCircleArea(C, p[i], p[i+1]);
            else ret += Cross(p[i+1] - C.c, p[i] - C.c) / 2.0;
        }
        if (cnt == 1)
        {
           if(InCircle(p[i],C)&&(!InCircle(p[i+1],C)||OnCircle(p[i+1],C)))ret += Cross(sol[0] - C.c, p[i] - C.c) / 2.0, ret += SegCircleArea(C, sol[0], p[i+1]);
            else ret += SegCircleArea(C, p[i], sol[0]), ret += Cross(p[i+1] - C.c, sol[0] - C.c) / 2.0;
        }
        if (cnt == 2)
        {
            if ((p[i] < p[i + 1]) ^ (sol[0] < sol[1])) swap(sol[0], sol[1]);
            ret += SegCircleArea(C, p[i], sol[0]);
            ret += Cross(sol[1] - C.c, sol[0] - C.c) / 2.0;
            ret += SegCircleArea(C, sol[1], p[i+1]);
        }
    }
    return fabs(ret);
}

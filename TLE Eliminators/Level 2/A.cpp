int sum_0_to_n(int n){
	if(n<=0)return 0;
	return sum_0_to_n(n-1)+n;
}
int f(int n){
	if(n<=1)return n;
	return f(n-1)+f(n-2);
}
int n;
void rec(string s){
	if(s.size()==n){
		cout<<s<<endl;
		return;
	}
	rec(s+"0");
	rec(s+"1");
	
}
int n;
string s;
void rec(){
	if(s.size()==n){
		cout<<s<<endl;
		return;
	}
	s.pb('0');
	rec();
	s.pop_back();
	
	s.pb('1');
	rec();
	s.pop_back();
	
}
class Solution{
	public:
		vi cur_subset;
		vvi final_output;
		
		void solve(int i,vi &nums){
			if(i==SZ(nums)){
				final_output.pb(cur_subset);
				return;
			}
			solve(i+1,nums);
			
			cur_subset.pb(nums[i]);
			solve(i+1,nums);
			cur_subset.pop_back();
		}
		vvi subsets(vi &nums){
			final_output.clear();
			solve(0,nums);
			return final_output;
		}
		
};

class Solution{
public:
	bool isValid(int i,int j,vector<vector<char>> &board,char c){
		REP(k,9)if(board[i][k]==c)return false;
		REP(k,9)if(board[k][j]==c)return false;
		for(int ki=i-i%3;ki<(i-i%3+3);ki++)
			for(int kj=j-j%3;kj<(j-j%3+3);kj++)
				if(board[ki][kj]==c)return false;
		
		return true;
	}
	bool solve(int i,int j,vector<vector<char>> &board){
		if(i==9)return true;
		if(j==9)return solve(i+1,0,board);
		if(board[i][j]!='.')return solve(i,j+1,board);
		for(char c='1';c<='9';c++){
			if(isValid(i,j,board,c)==false)continue;
			board[i][j] = c;
			bool result = solve(i,j+1,board);
			if(result==true)return true;
			board[i][j] = '.';
			
		}
		return false;
	}
	void solveSudoku(vector<vector<char>> &board){
		solve(0,0,board);
		
	}
};
bool isPowerOfFour(int n){
	if(n==1)return true;
	if(n==0)return false;
	if(n%4!=0)return false;
	return isPowerOfFour(n/4);
}

int helper(int i,int n,
vector<pii> &a,int w){
	if(i==0){
		if(w>=0)
			return 0;
		return INT_MIN;
	}
	int notTake = helper(i+1,n,a,w);
	int take = a[i].ss + helper(i+1,n,a,w-a[i].ff);
	return max(take,notTake);
}
class Solution{
public:
	bool helper(vector<vector<char<< &board,string
	word,int i,int j,int ind){
		if(ind>=SZ(word))return true;
		if(i<0 or i>=SZ(board) or j<0 or j>=SZ(board[0]) 
		or board[i][j]=='#' or word[ind]!=board[i][j])
			return false;
		char tmp = board[i][j];
		board[i][j]='#';
		REP(k,4){
			int ni,nj;
			ni= i + dx[k];
			nj = j+dy[k];
			if(helper(board,word,ni,nj,ind+1)==true)
				return true;
		}
		board[i][j]= tmp;
		return false;
	}
	bool exist(vector<vector<char>> &board,string word){
		int n,m;
		n = SZ(board);
		m = SZ(board[0]);
		REP(i,n){
			REP(j,m){
				if(helper(board,word,i,j,0)==true)
					return true;
			}
		}
		return false;
	}
};
int set_The_kth_bit(int n,int k){
	n = n | (1<<k);
	return n;
}
int check_The_kth_bit_set_or_not(int n,int k){
	return ((n & (1<<k))!=0);
}
int toggle_the_kth_bit(int n,int k){
	n = n^(1<<k);
	return n;

}
int unset_the_kth_bit(int n,int k){
	if(check_The_kth_bit_set_or_not(n,k)){
		n = toggle_the_kth_bit(n,k);
	}
	return n;
}
void reverse_sort(vi arr){
	sort(arr.rbegin(),arr.rend());
}
void comparator_Sort(vi arr){
	sort(ALL(arr),comparator());
}
void swap(int a,int b){
	a = a^b;
	b = a^b;
	a = a^b;
	
}
void bitmasking(){
	REP(mask,(1<<n)){
		REP(i,n){
			if(mask&(1<<i))
				cout<<a[i]<<" ";
		}
	}
	cout<<endl;
}
vi get_factors(int n){
	vi factors;
	for(int i=1;i*i<=n;i++){
		if(n%i==0)
		{
			factors.pb(i);
			if((n/i)!=i)		
			factors.pb(n/i);
		}
	}
	return factors;
}
vi prime_factorization_trial_division(int n){
	vi facts;
	for(int d=2;d*d<=n;d++){
		while(n%d==0){
			facts.pb(d);
			n/=d;
		}
	}
	if(n>1)
		facts.pb(n);
	return facts;
}
void sieve(int n){
	bool primes[n+1];
	fill(primes,primes+n+1,true);
	primes[0] = primes[1] = false;
	for(int i=2;i*i<=n;i++)
		if(primes[i])
			for(int j=i*i;j<=n;j+=i)
				primes[j]= false;
}
const int N = 1e7;
int spf[N],hpf[N];
void get_spf(){
	REP(i,N)
		spf[i]=i,hpf[f]=i;
	for(int i=2;i<N;i++){
		if(spf[i]==i){
			for(int j=2*i;j<N;j+=i){
				if(spf[j]==j)
					spf[j]= i;
			}
		}
		if(hpf[i]==i){
			for(int j=2*i;j<N;j+=i){
				hpf[j]=i;
			}
		}
	}
	vi prime_factors;
	while(n>1){
		prime_factors.pb(spf[n]);
		n/=spf[n];
	}
}
void lcm_of_array(vi &arr,int n){
	int temp = arr[0];
	for(int i=1;i<n;i++){
		temp = (temp*arr[i])/gcd(temp,arr[i]);
	}
	cout<<temp<<endl;
}
void nge_nse_pge_pse(vi &v,int n){
	vi ans(n);
	stack<int> st;
	for(int i=n-1;i>=0;i--){
		while(!st.empty() and st.top()<=v[i]
		//v[st.top()]<=v[i]
		)
		st.pop();
		if(st.empty()){
			ans[i] = -1;
			// if >= ans[i]=n;
		}else{
			ans[i] = st.top();
		}
		st.push(v[i]);
		// st.push(i);
	}
	for(auto it:ans)cout<<it<<" ";
}
// 2d prefix sum(usaco) and coordinate compression

void search(int n){
	int si=1,ei=n,ans=-1;
	while(si<=ei){
		int mid = si+(ei-si)/2;
		if(func(mid)){
			ans=mid;//last true
			ei=mid-1;
			//si=mid+1;
		}else{
			//ans=mid;//first false
			si=mid+1;
			//ei=mid-1;
		}
	}
}
int merge(vi &v,int si,int ei){
	vi arr(ei-si+1);
	int ans = 0;
	int mid = (si+ei)/2;
	int i=si,j=mid+1,k=0;
	while(i<=mid and j<=ei){
		if(v[i]<=v[j]){
			arr[k]=v[i];
			i++;k++;
		}else{
			ans+=(mid-i+1);
			arr[k]=v[j];
			j++;k++;
		}
	}
	while(i<=mid){
		arr[k]=v[i];
		i++;
		k++;
	}
	while(j<=ei){
		arr[k] = v[j];
		j++;
		k++;
	}
	int t=0;
	for(int r=si;r<=ei;r++){
		v[r] = arr[t];
		t++;
	}
	return ans;
}
int merge_sort(vi &v,int si,int ei){
	if(si>=ei)return 0;
	int mid = (si+ei)/2;
	int ans = 0;
	ans+=merge_sort(v,si,mid);
	ans+=merge_sort(v,mid+1,ei);
	ans+=merge(v,si,ei);
	return ans;
}
void coordinate_compression(){
	int n,x;
	cin>>n>>x;
	int a[n],b[n],c[n];
	REP(i,n)cin>>a[i]>>b[i]>>c[i];
	seti st;
	REP(i,n){
		st.insert(a[i]);
		st.insert(b[i]+1);
	}
	int idx = 0;
	map<int,int> mm;
	for(auto it:st){
		mm[it] = idx;
		idx++;
	}
	int compressed_size = idx;
	vi cost(compressed_size);
	REP(i,n){
		cost[mm[a[i]]]+=c[i];
		if(mm[b[i]+1]<compressed_size)
			cost[mm[b[i]+1]]-=c[i];
	}
	for(int i=1;i<compressed_size;i++){
		cost[i]+=cost[i-1];
	}
	int ans = 0;
	vi days(ALL(st));
	REP(i,compressed_size-1){
		int span = days[i+1] - days[i];
		ans+=min(cost[i],x)*span;
	}
	cout<<ans<<endl;
}
int main(){
	cin>>n;
	rec("");
	
	vi nums = {1,2,3,4};
	for(auto i:Solution().subset(nums)){
		for(auto j:i)
			cout<<j<<" ";
		cout<<endl;
	}
	int n,w;
	cin>>n>>w;
	vector<pii> a(n);
	REP(i,n){
		cin>>a[i].ff>>a[i].ss;
		
	}
	cout<<helper(0,n,a,w)<<endl;
}
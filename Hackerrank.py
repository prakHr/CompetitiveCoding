def encode(word):
    word=word[::-1]
    letters='abcdefghijklmnopqrstuvwxyz'
    ans=''
    for i in range(len(word)):
        idx=letters.index(word[i])
        ans+=letters[(idx+3)%len(letters)]
    return ans
def decode(word):
    word=word[::-1]
    letters='abcdefghijklmnopqrstuvwxyz'
    ans=''
    for i in range(len(word)):
        idx=letters.index(word[i])
        ans+=letters[(idx-3)%len(letters)]
    return ans
message='have a good day'
message=message.split(' ')
encoded=''
for i in range(len(message)):
    if i!=len(message)-1:
        encoded+=(encode(message[i]))+' '
    else:encoded+=(encode(message[i]))
print(encoded)
decoded=''
encoded_message=encoded.split(' ')
for i in range(len(encoded_message)):
    if i!=len(message)-1:
        decoded+=(decode(encoded_message[i]))+' '
    else:decoded+=(decode(encoded_message[i]))
print(decoded)

#https://www.hackerearth.com/practice/data-structures/arrays/1-d/practice-problems/5/?sort_by=partially%20solved&p_level=
import sys
def nthNumber(n):
    length=len(n)
    i=0
    ans=''
    digits='123456789'
    while(i<length):
        for d in range(len(digits)):
            if(i>=length):break
            if(n[i]==digits[d]):
                count=0
                while(i<length and n[i]==digits[d]):
                    count+=1
                    i+=1
                ans+=str(count)+digits[d]
                
        
    return ans
string='112'
print(nthNumber(string))
def PrefixAndSuffixWithMaxMinArr(arr):
    n=len(arr)
    prefixDPMax=[0]*n
    prefixDPMax[n-1]=arr[n-1]
    for i in range(n-2,-1,-1):
        prefixDPMax[i]=max(arr[i],prefixDPMax[i+1])

    prefixDPMin=[0]*n
    prefixDPMin[n-1]=arr[n-1]
    for i in range(n-2,-1,-1):
        prefixDPMin[i]=min(arr[i],prefixDPMin[i+1])

    suffixDPMax=[0]*n
    suffixDPMax[0]=arr[0]
    for i in range(1,n):
        suffixDPMax[i]=max(arr[i],suffixDPMax[i-1])

    suffixDPMin=[0]*n
    suffixDPMin[0]=arr[0]
    for i in range(1,n):
        suffixDPMin[i]=min(arr[i],suffixDPMin[i-1])
        
    return prefixDPMax,prefixDPMin,suffixDPMax,suffixDPMin
        
arr=[8,5,40,3,2,6]
prefixDPMax,prefixDPMin,suffixDPMax,suffixDPMin=PrefixAndSuffixWithMaxMinArr(arr)
print(suffixDPMin)
hasGreaterElementOnTheRight_in_O_n_time=[False]*len(arr)

for i in range(len(arr)-1):
    if arr[i]<prefixDPMax[i+1]:
        hasGreaterElementOnTheRight_in_O_n_time[i]=True
#print(hasGreaterElementOnTheRight_in_O_n_time)
#arr[i]<arr[k]<arr[j] for i<k<j
hasSmallerElementOnTheLeft_in_O_n_time=[False]*len(arr)
for j in range(1,len(arr)):
    if arr[j]>suffixDPMin[j-1]:
        print(arr[i],"=>",suffixDPMin[j-1])
        hasSmallerElementOnTheLeft_in_O_n_time[j]=True
print(hasSmallerElementOnTheLeft_in_O_n_time)

hasSmallerElementOnTheLeft=[False]*len(arr)

for j in range(1,len(arr)):
    for k in range(j-1):
        if arr[k]<arr[j]:
            hasSmallerElementOnTheLeft[j]=True
            break
print(hasSmallerElementOnTheLeft)

hasGreaterElementOnTheRight=[False]*len(arr)
for i in range(len(arr)):
    for k in range(i+1,len(arr)):
        if arr[k]>arr[i]:
            hasGreaterElementOnTheRight[i]=True
            break
#print(hasGreaterElementOnTheRight)

def merge(A,start,middle,end):
    L=A[start:middle+1]
    R=A[middle+1:end]
    L.append(sys.maxsize)
    R.append(sys.maxsize)
    i=j=0
    for k in range(start,end+1):
        if L[i]<=R[j]:
            A[k]=L[i]
            i+=1
        else:
            A[k]=R[j]
            j+=1
    
def merge_sort_util(A,start,end):
    if start<end:
        middle=(start+end)//2
        merge_sort_util(A,start,middle)
        merge_sort_util(A,middle+1,end)
        merge(A,start,middle,end)
        
def merge_sort(A):
    merge_sort_util(A,0,len(A)-1)
    
def insertionSort(arr):
    for i in range(1,len(arr)):
        currentValue=arr[i]
        position=i
        while position>0 and arr[position-1]>currentValue:
            arr[position]=arr[position-1]
            position=position-1
        arr[position]=currentVal
#bubble sort
def bubbleSort(arr):
    if len(arr)<=1:
        return arr
    for k in range(len(arr)):
        for i in range(len(arr)-1):
            if arr[i+1]<arr[i]:
                arr[i],arr[i+1]=arr[i+1],arr[i]
    return arr

#Selection sort
def selectionSort(arr):
    if len(arr)<=1:
        return arr
    for i in range(len(arr)-1):
        for j in range(i+1,len(arr)):
            if arr[i]>arr[j]:
                arr[i],arr[j]=arr[j],arr[i]
    return arr
arr1=[145,180,165,170,180,176,180,134,176]
arr2=[6,5,4,3,2,1,0,0,-1]
arr1=selectionSort(arr1)
arr2=bubbleSort(arr2)
print(arr2)
#Check if num Power of four
def isPower(num,y):
    i=1
    flag=False

    while True:
        x=y**i
        if x>=num:
            if x==num:
                flag=True
            break
        i=i+1
    return flag        
numbers=[4,4**2,4**3,63,22,65]
y=4
for num in numbers:
    print(isPower(num,y))


#Array Manipulation
#nstead of storing the actual values in the array, you store the difference between the current element and the previous element. So you add sum to a[p] showing that a[p] is greater than
#its previous element by sum. You subtract sum from a[q+1] to show that a[q+1] is less than a[q] by sum (since a[q] was the last element that was added to sum). By the end of all this,
#you have an array that shows the difference between every successive element. By adding all the positive differences, you get the value of the maximum element
n,k=map(int,input().split())
arr=[0]*(1+n)
for i in range(k):
    l,r,val=map(int,input().split())#Lets assume we add  this val to the entire array then we subtract this val from a[q+1] onwards .We need to keep track of a[p] and a[q+1] only
    arr[l]+=val
    if(r+1<=n):
        arr[r+1]-=val
max=0
for i in range(1,1+n):
    x+=arr[i]
    max=max(x,max)
print(max)
     
from collections import Counter
import sys
sys.setrecursionlimit(10000)
T=int(input())
for _ in range(T):
    N,K=map(int,input().split())
    S=list(sorted(map(int,input().split())))
    A=[0]*N
    used=Counter(S)
    i=0
    for s in S:
        if used[s]==0:
            continue
        A[i]=s-(K-1)*A[0] if i!=0 else s//K
        
        def set_used(sum,idx,cnt):
            if cnt==K:
                used.subtract([sum])
                return
            set_used(sum+A[idx],idx,cnt+1)
            if idx>0:
                set_used(sum,idx-1,cnt)

        set_used(A[i],i,1)
        i+=1
        if i==N:
            break
    print(' '.join(map(str,A)))
            
#Longest Mod Path
#maximum value of a+bk mod M,where a=vsimple,b=vcycle
#when gcd(b,M)=1 value can be M-1
#else gcd(b,M)=g a=dis[e-1]-dist[s-1] value=m-g+a%g

n=input()
adj=[[]for i in range(n)]
for i in range(n):
    a,b,c=map(int,input().split())
    adj[a-1].append((b-1,c))
    adj[b-1].append((a-1,-c))
dist=[None]*n
parents=[set() for i in range(n)]
dist[0]=0
stack=[0]
while stack:
    i=stack.pop()
    for j,c in adj[i]:
        if j in parents[i]:
            continue
        ndist=dist[i]+c
        parents[j].add(i)
        if dist[j] is None:
            dist[j]=ndist
            stack.append(j)
        else:
            cycle=abs(dist[j]-ndist)
from fractions import gcd
for q in range(input()):
    s,e,m=map(int,input().split())
    a=gcd(cycle,m)
    print (m-a+(dist[e-1]-dist[s-1])%a)
    
#Turn off the Lights
n,k=map(int,input().split())

ans=10**13
Cost=list(map(int,input().split()))
for bulb in range(1,1+(1+k)):
    cost=0
    next=bulb
    while next<=n:
        cost+=Cost[next]
        next+=(2*k+1)
    next-=(2*k-1)
    if(next+k>=n):
        ans=min(cost,ans)
print(ans)
#Mandragora Forest
T=int(input())
for _ in range(T):
    N=int(input())
    H=list(map(int,input().split()))
    x=sum(H)#if battles all,with S=1 ,P=S*H[i]=1*sum(H)
    S,var=1,0
    B=sorted(H)
    P=x
    for i in range(N):
        S+=1
        x=x-B[i]#then keep eating till a point
        var=S*x
        if var>P:
            P=var
    print(P)
#Summing Pieces
n=10**6+3
s_array=[1]
mod=10**9+7
for i in range(1,n):
    s_array.append((s_array[i-1]*2)%mod)
size=input()
array=list(map(int,input().split()))
m_array=[0]*size
start=size-2
m_array[0]=m_array[-1]=(s_array[size]-1+mod)%mod
for i in range(1,size//2+1):
    m_array[i]=(s_array[start]-s_array[i-1]+mod)%mod
    m_array[i]=(m_array[i-1]+m_array[i])%mod
    m_array[-i-1]=m_array[i]
    start=start-1
ans=0
for i in range(size):
    ans=(ans+(m_array[i]*((array[i])%mod))%mod)%mod
print (ans)


#Xor and Sum
def getBit(x,i):
    if i<len(x):
        return int(x[i])
    return 0
def calcsumxor(a,b):
    C=314159
    MOD=10**9+7
    a=a[::-1]
    b=b[::-1]
    #print(a,b)
    p=1
    n=len(a)
    m=len(b)
    count0=C
    count1=0
    s=0#ans is sigma i=0->C(a xor (b shl i)) (10 xor(1010 shl i))
    #a=01
    #b=0101,now do bit by bit computation from higher bits that's why reverse
    for i in range(max(n,m+C)):#range is in accordance to (b shl C)
        if getBit(b,i)==1:#Now iterate over ith bit of b to see if it is 1/0
            count1=count1+1
        else:
            count0=count0+1
            #Then update count0/1
        if getBit(a,i)==1:##Now iterate over ith bit of a to see if it is 1/0
            s=(s+count0*p)%MOD
        else:
            s=(s+count1*p)%MOD
        p=(p*2)%MOD
        if i>=C:
            if getBit(b,i-C)==0:
                count0=count0-1
            else:
                count1=count1-1
        else:
            count0=count0-1
        count0=count0%MOD
        count1=count1%MOD
    return s
a=input()
b=input()
print(calcsumxor(a,b))
            



#Candies
n=int(input())
childs=[0]*n
for i in range(0,n):
    t=int(input())
    childs[i]=t
a=[1]*n
for i in range(1,n):
    if childs[i]>childs[i-1]:
        a[i]=a[i-1]+1
    else:
        a[i]=1
for i in range(n-1,0,-1):
    if childs[i-1]>childs[i]:
        a[i-1]=max(a[i-1],a[i]+1)
print(sum(a))
#Prime XOR
from collections import Counter

def F(a):
    prime=[True]*(8192)
    prime[0]=False
    prime[1]=False
    for d in range(2,4096):
        if prime[d]:
            k=2*d
            while k<8192:
                prime[k]=False
                k+=d
    mod=10**9+7
    #Counter can take anything like a list,dictionary etc
    #Counter will create a dictionary feel with key as a form of counts of values
    cnt=Counter(a)
    #####print(cnt)
    a=list(set(a))
    #print(a)
    n=len(a)
    dp=[[0]*8192 for _ in range(n+1)]
    dp[0][0]=1
    for i in range(1,n+1):
        for x in range(8192):
            #divided into 2 parts odd and even counts for every x
            dp[i][x]=dp[i-1][x]*((cnt[a[i-1]]+2)//2)+dp[i-1][x^a[i-1]]*((cnt[a[i-1]]+1)//2)
            dp[i][x]%=mod
    res=0
    for x in range(8192):
        #important part,only prime[x] is required for ans
        if prime[x]:
            res+=dp[n][x]
            res%=mod
    return res

T=int(input())
for _ in range(T):
    n=int(input())
    a=list(map(int,input().split()))
    print(F(a))
#Modified Fibonacci
a=[0]*20
b=list(map(int,input().split()))
a[0]=b[0]
a[1]=b[1]
n=b[2]
for i in range(2,n):
    a[i]=a[i-2]+a[i-1]*a[i-1]
print(a[n-1])

#Sherlock and Cost
T=int(input())
for _ in range(T):
    N=int(input())
    B=[int(x) for x in input().split()]
    hi,low=0,0
    for i in range(1,N):
        high_to_low_diff=abs(B[i-1]-1)
        low_to_high_diff=abs(B[i]-1)
        #high_to_high_diff=abs(B[i]-B[i-1])
        
        low_next=max(low,hi + high_to_low_diff)
        hi_next=max(hi,low + low_to_high_diff)

        low=low_next
        hi=hi_next
    x=max(hi,low)
    print(x)
        

#Nim Game
T=int(input())
for _ in range(T):
    N=int(input())
    a=[int(x) for x in input().split()]
    ans=a[0]
    for i in range(1,N):
        ans=ans^a[i]
    if ans==0:
        print("Second")
    else:
        print("First")
        

#Chessboard Engine
def call(x,y):
    if 1<=x<=15 and 1<=y<=15:
        return not call(x-2,y+1) or not call(x-2,y-1) or not call(x+1,y-2) or not call(x-1,y-2)
    else:
        return True
t=int(input())
for _ in range(t):
    x,y=map(int,input().split())
    if call(x,y)==0:
        print("Second")
    else:
        print("First")


#Game(of Stones) Theory
dp=[0]*200
dp[0]=0
dp[1]=0
dp[2]=1
dp[3]=1
dp[4]=1
dp[5]=1
for i in range(6,103+1):
    # If no possible move results in the next state
    #being a losing position, then the player loses.
    #In other words, if a player cannot make a move from their
    #current state that results in the next state being a losing position,
    #the player is in a losing position! 
    if not(dp[i-2] and dp[i-3] and dp[i-5]):
           dp[i]=1
    else:
           dp[i]=0
t=int(input())
while t>0:
    t=t-1
    n=int(input())
    if dp[n]:
        print("First")
    else:
        print("Second")
           


#Superdigit Sum
a=list(map(str,input().split()))
def super_digit(n):
    sum=0
    if len(n)==1:
        print(n)
    else:
        a=len(n)
        for i in range(a):
            sum+=int(n[i])
        super_digit(str(sum))
sum=0
a1=len(a[0])
for i in range(a1):
    sum+=int(a[0][i])
n=sum*int(a[1])
c=super_digit(str(n))
print(c)

#Reconstruction using Algebraic Expressions
n=int(input())
a=[int(x)for x in input().strip().split()]
visited=[[False]*(101) for x in range(n)]
visited[0][a[0]]=True
for i in range(1,n):
    for v in range(101):
        if visited[i-1][v]:
            visited[i][(v+a[i])%101]=True
            visited[i][(v-a[i])%101]=True
            visited[i][(v*a[i])%101]=True
            
v=0
for i in range(n-1,0,-1):
    for w in range(101):
        if visited[i-1][w]:
            if(w+a[i])%101==v:          
                a[i]='+'+str(a[i])
                v=w
                break
            if(w-a[i])%101==v:          
                a[i]='-'+str(a[i])
                v=w
                break
            if(w*a[i])%101==v:          
                a[i]='*'+str(a[i])
                v=w
                break
print(*a,sep='')


def xor(arr):
        ans=0
        for i in range(len(arr)):
                ans=ans^arr[i]
        return ans
T=int(input())
for _ in range(T):
        n=int(input())
        elements=list(map(int,input().split()))
        subsets=[]
        for i in range(1<<n):
                a=[]
                for j in range(n):
                        if(i&(1<<j)):
                                a.append(elements[j])
                if a==[]:
                        continue
                subsets.append(xor(a))
        print(xor(subsets))

marks=[82,95,85,91,79]
print(sum(marks))
x=sum(marks)/len(marks)
print(x)
def titleToNumber( s):
        """
        :type s: str
        :rtype: int
        """
        letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        m={}
        for i,l in enumerate(letters):
            m[l]=i+1
        N=len(s)
        ans=0
        power=0
        s=s[::-1]
        for i in range(N):
                ans=pow(26,i)*m[s[i]]+ans
        return ans
print(titleToNumber("AB"))
def imageSmoother( M):
        """
        :type M: List[List[int]]
        :rtype: List[List[int]]
        """
        x=len(M[0])
        y=len(M)
        newList=[]
        newList.append([0]*(x+2))
        for a in M:
            aList=[0]+a+[0]
            newList.append(aList)
        newList.append([0]*(x+2))
        for n in newList:
                print(n)
        return newList
M=[[1,1,1],
 [1,0,1],
 [1,1,1]]
(imageSmoother( M))
roman={1:"I",5:"V",10:"X",50:"L",100:"C",500:"D",1000:"M"}
inverse_roman={}
for k,v in roman.items():
        inverse_roman[v]=k
print(inverse_roman)
def firstUniqChar( s: str) -> int:
        m={}
        for ss in s:
            if ss not in m:
                m[ss]=0
            m[ss]+=1
        print(m)    
        for char in s:
            if m[char]==0:
                return char
        return -1
print(firstUniqChar("leetcode"))
def compress( chars):
        pointer=0
        a=[]
        for i in range(len(chars)):
            if chars[i]!=chars[pointer]:
                a=a+[chars[pointer],str(i-pointer)]
                pointer=i
        a=a+[chars[pointer],str(i-pointer+1)]
        chars=a
        return len(chars)
print(compress(["a","a","b","b","c"]))
def helper( s):
        ans = ''
        pointer = 0
        for i in range(len(s)):
            if s[pointer] != s[i]:
                k = str(i-pointer) + s[pointer]
                ans += k
                pointer = i
        ans += str(len(s)-pointer)+s[len(s)-1]
        return ans
print(helper("121"))
        
letters='abcdefghijklmnopqrstuvwxyz'
m={}
for l in letters:
    m[l]=0
def equalLandR(s):
    return s.count('L')==s.count('R')
def balancedStringSplit(s):
    count=0
    current_string=s[0]
    for i in range(1,len(s)):
        current_string+=s[i]
        if equalLandR(current_string):
            current_string=''
            count+=1
    return count
print(balancedStringSplit("RLRRRLLRLL"))
            
def addToArrayForm(A, K):
        n=len(A)
        first_digit=A[n-1]
        ans=[]
        initial_sum=first_digit+K
        ans.append(initial_sum%10)
        carry=initial_sum//10
        
        for i in range(2,n+1):
            initial_sum=A[n-i]+carry
            ans.append(initial_sum%10)
            carry=initial_sum//10
        if(carry):
            ans.append(carry)
        return ans[::-1]
print(addToArrayForm([9,9,9,9,9,9,9,9,9,9],1)) 
def binaryBits(n):
    if n==0:
        return ['0']
    if n==1:
        return ['0','1']
    res=binaryBits(n-1)
    return ['0' + s for s in res]+['1' + s for s in res[::-1]]
#Binary to gray abcd=>efgh where e=a,f=a^b,g=b^c,h=c^d
#gray to Binary efgh=>abcd where a=e,b=e^f,c=e^f^g,d=e^f^g^h therefore G=B^(B>>1)
print([int(s,2) for s in binaryBits(2)])
def integerbinaryBits(n):
    res=[]
    for i in range(pow(2,n)):
        res.append(i^(i>>1))
    return res

string_input='aacccddeeeeAACCCCCC'
alphabets_map={}
for s in string_input:
    key=s
    alphabets_map[key]=alphabets_map.get(key,0)+1
#print(alphabets_map)
my_list=[]
for key,value in alphabets_map.items():
    my_list.append([value,key])
my_list=sorted(my_list)
my_list=(my_list[::-1])
ans=''
for a in my_list:
    ans=ans+a[0]*a[1]
print(ans)

def hasAlternatingBits( n: int) -> bool:
    prev=0
    current=n%2
    while(n>0):
        prev=current
        n=n//2
        current=n%2
        if(prev==current):
            return False
    return True

arr=[11,10]

for a in arr:
    print(hasAlternatingBits(a))
import numpy as np
predicted_values=[]
predicted_values.append(np.array([1,2,3]))
predicted_values.append(np.array([1,2,3]))


print(predicted_values)
prediction_class=np.array([[0.78],[0.67],[0.43]])
prediction_class=prediction_class.ravel()
prediction_class=np.reshape(prediction_class,(-1,1))
print(prediction_class)

hh=['a','bb','ccc']
for h in hh:
    print(hh.index(h))
    
def productSum(arr):#arr=[a,b,c]
    #(a+b+c)^2=a^2+b^2+c^2+2*ans
    if len(arr)<=1:return 0
    squared_arr=[a**2 for a in arr]
    squared_arr_sum=sum(squared_arr)
    sum_squared=(sum(arr))**2
    return (sum_squared-squared_arr_sum)/2
print("Product sum of an array is")
print(productSum(arr=[1, 3, 4]))

#To print 3 larger no
import sys#64 bit(1<<63 -1)

INT_MAX=sys.maxsize
INT_MIN=-1*INT_MAX
def ThreelargestInO_n_time(arr,INT_MIN):
    if len(arr)<3:
        return -1
    first=second=third=INT_MIN
    for i in range(len(arr)):
        if first<arr[i]:
            third=second
            second=first
            first=arr[i]
        elif second<arr[i]<first:
            third=second
            second=arr[i]
        elif third<arr[i]<second:#if this is case then we dont need to set the first and second value only third largest no is going to be arr[i]
            third=arr[i]
    return first,second,third
arr=[1,0.1,9,11,77]
a,b,c=ThreelargestInO_n_time(arr,INT_MIN)
print("Three largest no are")
print(a,",",b,",",c)
#To print 2 small numbers(Can be extended to print three larger number)

arr=[1,0.1,9,11,77]
N=len(arr)
smaller=INT_MAX#we dont know next_small element so assume it 31 bit no.
smallest=arr[0]#we assume smallest element is first element
for i in range(1,N):
    if smallest<arr[i]<smaller:
        smaller=arr[i]
        
    elif arr[i]<smallest:
        smaller=smallest
        smallest=arr[i]
print("Two smallest no are")    
print(smallest," and ",smaller)
        

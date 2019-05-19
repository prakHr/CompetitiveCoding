
#1145A
m=n=int(input())
a=list(map(int,input().split())
#while all(a[i:i+m]!=sorted(a[i:i+m]) for i in range(0,n,m)):
 #  m=m//2
   
#print(m)
#1151A
#input()
#s=input()
#print(min(sum(min(x,26-x) for x in map(lambda x,y:abs(ord(x)-ord(y)),t,'ACTG') for t in zip(s,
 #                                                                                           s[1:],s[2:],s[3:]))))
#46A
n=int(input())
C=0
for i in range(n-1):
    C=(C+i+1)%n
    print(C+1,' ')
    
#47A
n=int(input())
print('Yneos'[(8*n+1)**0.5%1>0::2])
#48A
n,v=['F','M','S'],[['rock','paper','scissors'].index(input()) for i in range(3)]
for i in range(3):
    if v[(i-1)%3]==(v[i]-1)%3 and v[(i+1)%3]==(v[i]-1)%3:
        print(n[i])
        exit()
print('?')
#49A
question=input()
lastLetter=''
vowels=['A','E','I','O','U','Y','a','e','i','o','u','y']
if question[-2]==' ':
    lastLetter=question[-3]
else:
    lastLetter=question[-2]
if lastLetter in vowels:
    print('YES')
else:print('NO')
#52A
n=int(input())
s=input()
x=n-max(s.count('3'),s.count('2'),s.count('1'))

#53A
s=input()
n=int(input())
l=[]
for i in range(n):
    d=input()
    if d[:len(s)]==s:
        l.append(d)
if len(l)>0:
    print(min(l))
else:
    print(s)
aBitAboutYourself='''
I am a student at bits hyderabad campus who loves programming, watching sports, listening to music etc.

'''
#40A
import math
x,y=map(int,input().split())
r1=math.floor(sqrt(x*x+y*y))
r2=math.ceil(sqrt(x*x+y*y))
if r1==r2:
    if r1%2==0:
        print('black')
    else:
        print('white')
else:
    if r2%2==0:
        print('black')
    else:
        print('white')
        
#43A
n=int(input())
a=set()
twoTeams=[]
for _ in range(n):
    aa=input()
    twoTeams.append(aa)
    a.add(aa)
a=list(a)
first,second=a[0],a[1]
print(first,second)
    
    
#26A(Not understood)
n=int(input())
countt=0
for j in range(2,n):
    count=0
    for i in range(2,n):
        if n%i==0 and isPrime(i):
            count+=1
    if count==2:
        countt+=1
print(countt)   
        
#27A
n=int(input())
index=list(map(int,input().split()))
for i in range(1,3001):
    if i not in index:
        exit(print(i))
#29A
n=int(input())
sumSet=[]
X,D=[],[]
for _ in range(n):
    x,d=map(int,input().split())
    if [x+d,x] in l:
        print('YES')
        exit()
    l.append([x,x+d])

    #i,j=>(xi+di,dj)(xj+dj,di)=>xj=xi+di,xj+dj=xi

print('NO')
#25A
count1,count2=0,0
n=int(input())
l=[int(x)%2 for x in input().split()]
print(l.index(sum(l)==1)+1)
print(sum(l)==1)
print(l.index(sum(l)==1))
            
            
            
#16A
n,m=map(int,input().split())
prev='a'
flag=True
for i in range(n):
    s=input()
    if (s.count(s[0])==m and s[0]!=prev and flag):
        prev=s[0]
        continue
    flag=False

#13A
def baseDigitSum(num,base):
    d=0
    while num!=0:
        d+=(num%base)
        num=num//base
    return d
A=int(input())
ans=0
for base in range(2,A):
    ans+=baseDigitSum(A,base)
ans1=str(ans)
ans2=str(A-2)
print(ans1+"/"+ans2)

#12A
line1=input()
line2=input()
line3=input()
flag=True
a,b,c=line1[0],line1[1],line1[2]       
aa,bb,cc=line2[0],line2[1],line2[2]
aaa,bbb,ccc=line3[0],line3[1],line3[2]
if a=='X' and ccc!='X':
    flag=False
if b=='X' and bbb!='X':
    flag=False
if c=='X' and aaa!='X':
    flag=False
if aa=='X' and cc!='X':
    flag=False
if cc=='X' and aa!='X':
    flag=False
if aaa=='X' and c!='X':
    flag=False
if bbb=='X' and b!='X':
    flag=False
if ccc=='X' and a!='X':
    flag=False
if flag==True:print("YES")
else:print("NO")
    
#8A
import re
s=input()
p=input()+'.*'+input()
a=re.search(p,s)
b=re.search(p,s[::-1])
ans=''
if a and b:
    ans='both'
elif a:
    ans='forward'
print(ans)

    
    

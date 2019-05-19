#1130A
k=l=(int(input())-1)//2
for x in map(int,input().split()):k-=x>0;l-=x<0
print((0,1,-1,1)[(k<0)+2*(l<0)])

#1132A
a,b,c,d=(int(input())for _ in[0]*4)
print(int(a==d and(a>0 or c==0)))
#1133A
t=0
for _ in 0,0:h,m=map(int,input().split(':'));t+=60*h+m
print(f'{t//120:02}:{t//2%60:02}')
#1139A
n=int(input())
l=input()
c=0
for i in range(0,n):
    if int(l[i])%2==0:
        c+=i+1
print(c)

#1140A
input()
r=k=0
i=1
for x in map(int,input().split()):
    k=max(k,x)
    r+=k==i
    i+=1
print(r)
#1141A
n,m=map(int,input().split())
r=-1
if m%n==0:
    m=m//n
    r=0
    for d in 2,3:
        while m%d==0:
            m=m//d
            r+=1
print((r,-1)[m>1])
#1144A
#the number of letters between the letter with the maximum alphabet position
#and the letter with the minimum alphabet position plus one is exactly the length of the string
 
for _ in [0]*int(input()):
    a=sorted(map(ord,input()))
    print('YNeos'[a!=[*range(a[0],1+a[-1])]::2])

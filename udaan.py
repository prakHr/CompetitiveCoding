'''
# Sample code to perform I/O:

name = input()                  # Reading input from STDIN
print('Hi, %s.' % name)         # Writing output to STDOUT

# Warning: Printing unwanted or ill-formatted data to output will cause the test cases to fail
'''
def addScreen(screen_name,noOfRows,TotalSeatsPerRow,ListOfAisleSeats,AISLE):
    array=[]
    for i in range(noOfRows):
        b=[]
        for j in range(TotalSeatsPerRow):
            b.append(EMPTY_SEATS)
        array.append(b)
    
    for t in range(len(ListOfAisleSeats)):
        x=ListOfAisleSeats[t]-1
        for i in range(noOfRows):
            if(x>=TotalSeatsPerRow or x<0):
                return('failure') 
            
    for t in range(len(ListOfAisleSeats)):
        x=ListOfAisleSeats[t]-1
        for i in range(noOfRows):
            array[i][x]=AISLE
    
    return ('success',array)

def ReserveSeats(getArray,row_number,ListOfReservedSeats,RESERVED):
    for j in len(getArray[0]):
        for l in ListOfReservedSeats:
            if not (0<=l<len(getArray[0])):
                return('failure')
                
    for j in len(getArray[0]):
        for l in ListOfReservedSeats:
            if j==l:
                getArray[row_number-1][j]=RESERVED
    return('success',getArray)
    
def getUnreservedSeats(array,row_no,RESERVED):
    ans=[]
    for j in len(array[0]):
        if array[row_no][j]!=RESERVED:
            ans.append(j+1)
    return(ans)

def SuggestContiguousSeats(getArray,row_no,noOfSeats,choice_of_seat_no,RESERVED,AISLE):
    c=len(getArray[0])
    ans=[]
    if(choice_of_seat_no<0 or choice_of_seat_no>=c):
        return(ans)
    num=noOfSeats
    start=choice_of_seat_no
    left=start-1
    while(getArray[row_no][left]!=RESERVED):
        ans.append(left)
        left-=1
        num=num-1
        if(left==0):
            break
        if(num==0):
            return(ans)
            
    right=start+1
    while(getArray[row_no][right]!=RESERVED):
        ans.append(right)
        right+=1
        num=num-1
        if(right==c):
            break
        if(num==0):
            return ans
    if(len(ans)==noOfSeats):
        return(ans)
    a=[]
    return(a)
# Write your code here
t=int(input())
data={}
AISLE=2
EMPTY_SEATS=0
RESERVED=1
for i in range(t):
    my_input=input()
    my_str_input=str(my_input)
    if(my_str_input[:10]=='add-screen'):
        remaining_str=my_str_input[11:].split()
        screen_name,noOfRows,TotalSeatsPerRow=remaining_str[0],int(remaining_str[1]),int(remaining_str[2])
        ListOfAisleSeats=[]
        
        for i in range(3,len(remaining_str)):
            ListOfAisleSeats.append(int(remaining_str[i]))
        someInputs=addScreen(screen_name,noOfRows,TotalSeatsPerRow,ListOfAisleSeats,AISLE)
        if(someInputs[0]=='f'):
            print('failure')
        else:
            print(someInputs[0])
            data[screen_name]=someInputs[1] 
    elif(my_str_input[:12]=='reserve-seat'):
        remaining_str=my_str_input[13:].split()
        screen_name,row_number=remaining_str[0],int(remaining_str[1])
        ListOfReservedSeats=[]
        
        for i in range(2,len(remaining_str)):
            ListOfReservedSeats.append(int(remaining_str[i])-1)
        #ReserveSeats(screen_name,row_number,ListOfReservedSeats)
        if screen_name not in data:
            print('failure')
            continue
        else:
            row_number=row_number-1
            getArray=data[screen_name]
            row_size=len(getArray)
            if(row_number<0 or row_number>=row_size):
                print('failure')
                continue
            if(0<=row_number<row_size):
                someInputs=ReserveSeats(getArray,row_number,ListOfReservedSeats,RESERVED)
                if(someInputs[0]=='f'):
                    print('failure')
                    continue
                else:
                    print(someInputs[0])
                    data[screen_name]=getArray
    elif(my_str_input[:len('get-unreserved-seats')]=='get-unreserved-seats'):
        remaining_str=my_str_input[len('get-unreserved-seats')+1:].split()
        screen_name,row_no=remaining_str[0],int(remaining_str[1])
        if screen_name not in data:
            print()
            continue
        else:
            array=data[screen_name]
            r=len(array)
            if(row_no<0 or row_no>=r):
                print()
                continue
            List=getUnreservedSeats(array,row_no-1)
            print(List)
        #continue
    elif(my_str_input[:len('suggest-contiguous-seats')]=='suggest-contiguous-seats'):
        remaining_str=my_str_input[len('suggest-contiguous-seats')+1:].split()
        screen_name,noOfSeats,row_no,choice_of_seat_no=remaining_str[0],int(remaining_str[1]),int(remaining_str[2]),int(remaining_str[3])
        if screen_name not in data:
            print()
            continue
        else:
            getArray=data[screen_name]
            r=len(getArray)
            if(row_no<0 or row_no>=r):
                print()
                continue
            List=SuggestContiguousSeats(getArray,row_no-1,noOfSeats,choice_of_seat_no-1,RESERVED,AISLE)
            print(List)
        #continue
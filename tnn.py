import numpy as np

r'''
function [b,c,i]=dqd2(b,c);

% takes 2 lower bidiagonal matrices with offdiagonal elements b and c
% and returns BD((I+diag(b,-1))*(I+diag(c,-1))
% if i>0 then c(i)=0 became positive and may need to be chased further

n=length(b);

t=c(1);
c(1)=b(1)+c(1);
d=b(1);
b(1)=0;

i=1;
while (i<n)&(b(i+1)~=0) 
   e=b(i+1)/c(i);
   d=e*d;
   b(i+1)=e*t;
   t=c(i+1);
   c(i+1)=c(i+1)+d;
   i=i+1;
end
'''

def dqd2(b, c):
    flag = False
    # Subroutine for 4.1
    if(b[0] == 0):
        return 0
    t = c[0]
    c[0] = b[0] + c[0]
    d = b[0]
    b[0] = 0
    i = 0
    while((i < len(b) - 1) and (b[i+1] != 0)):
        if(c[i] == 0):
            flag = True
        e = b[i+1]/c[i]
        d = e * d
        b[i+1] = e * t
        t = c[i+1]
        c[i+1] = c[i+1] + d
        i += 1
    if(flag):
        print(b)
        print(c)
        exit()
    return i

r'''
% function B=TNAddToNext(B,x,i)
%
% Computes BD(E_i(x)*A) where B=BD(A); i>=2
% In other words, given the bidiagonal decomposition B of a matrix 
% A, computes the bidiagonal decomposition of C, where C is obtained from 
% A by adding a multiple x of row i-1 to i
%
% Copyright (c) 2004 Plamen Koev. See COPYRIGHT.TXT for more details.
%
% Written October 19, 2004

function B=TNAddToNext(B,x,i)

[m,n]=size(B);

z=0; 
while (z<min(i-1,n)) & ((z==0) | (B(i-1,z)==0))
   % put the appropriate diagonals of B in b and c

   b=zeros(m-i+1,1);
   c=zeros(m-i+1,1);
   for j=0:m-i
      if z+j>0
         if z+j<=n, 
            b(j+1)=B(j+i,z+j);
         end
      else b(j+1)=x;   
      end
      if z+j+1<=n
         c(j+1)=B(j+i,z+j+1);
      end
   end

   [b,c,q]=dqd2(b,c); % q=0 means no nonzeros in c were created
                      % if nonzeros were created in c, the new nonzeros may have to be chased

   % return the new b and c in the same diagonals of B
   for j=0:m-i
      if (z+j>0) & (z+j<=n), B(j+i,z+j) =b(j+1); end
      if (z+j+1<=n), B(j+i,z+j+1) =c(j+1); end
   end
      
   i=i+q-1;
   z=z+q;
end
'''

def bd_multiply(bd, x, i):
    '''
    Find the BD decomposition for A * e_i(x), where bd is the decomposition for A.
    Changes to account for 0-indexing
    Assumes that i is 1-indexed
    '''
    m, n = bd.shape
    z = 0
    #print(bd, x, i)
    while((z < min(i-1, n)) and (z == 0 or bd[i-1-1, z-1] == 0)):
        b = np.zeros(m-i+1)
        c = np.zeros(m-i+1)
        for j in range(m-i+1):
            if(z+j > 0):
                if(z+j <= n):
                    b[j+1-1] = bd[j+i-1, z+j-1]
            else:
                b[j+1-1] = x
            if(z+j+1 <= n):
                c[j+1-1] = bd[j+i-1, z+j+1-1]
        q = dqd2(b, c)
        for j in range(m-i+1):
            if (z+j > 0) and (z+j <= n):
                bd[j+i-1, z+j-1] = b[j+1-1]
            if (z+j+1 <= n):
                bd[j+i-1, z+j+1-1] = c[j+1-1]
        i += q-1
        z += q

r'''
function B=TNSubmatrix(B,i)

% function B=TNSubmatrix(B,i);
%
% if B=BD(A), computes BD(A([1:i-1,i+1:m],:)), i.e. erases row i in A
% 
% Written, September 28, 2004
% Copyright (c) 2004 Plamen Koev. See COPYRIGHT.TXT for more details.

[m,n]=size(B);
if i<m
   for j=1:min(i-1,n)
      B(j+1:m,j+1:n)=TNAddToNext(B(j+1:m,j+1:n),B(i+1,j),i-j+1);
      B(i+1,j)=B(i+1,j)*B(i,j); 
   end
   
   for j=min(n,m)+(m>n):-1:i+1
      B(j,j-1)=B(j,j-1)*B(j-1,j-1);
      if (j<=n), B(j,j)=B(j,j)/B(j,j-1); end
   end
   
   for j=i+1:n
      B(i+1:m,i:n)=TNAddToNext(B(i+1:m,i:n)',B(i,j),j-i+1)';
   end
end

B=B([1:i-1,i+1:m],:);
'''

def remove_row(bd, ind):
    m, n = bd.shape
    ind += 1 # 1-indexing
    if(ind < m):
        for j in range(1, min(ind-1, n)+1):
            bd_multiply(bd[j+1-1:, j+1-1:], bd[ind+1-1,j-1], ind-j+1)
            bd[ind+1-1, j-1] *= bd[ind-1, j-1]
        for j in range(min(n,m)+(m>n), ind, -1):
            bd[j-1, j-1-1] *= bd[j-1-1, j-1-1]
            if(j <= n):
                bd[j-1, j-1] /= bd[j-1, j-1-1]
        for j in range(ind+1, n+1):
            bd_multiply(bd[ind+1-1:,ind-1:].T, bd[ind-1, j-1], j-ind+1)
    return np.delete(bd, ind-1, 0)

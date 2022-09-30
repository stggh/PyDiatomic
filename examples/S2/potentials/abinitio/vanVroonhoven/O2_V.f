      subroutine O2_V (R, i, E)

      implicit none

      integer i
      real*8 R, E

      include 'O2_V_data.h'

      goto (100, 200, 300, 400, 500, 600, 700, 800) i
      write (*,*)  '** ERROR in O2_V: i not between 1 and 8, i =', i
      stop 230
100   continue
      call O2_VV(R,nmax(1),alpha(1),beta(1),alphap(1),cp(1),rp(1),
     +           ca_1, c2a_1, clr_1, E)
      return
200   continue
      call O2_VV(R,nmax(2),alpha(2),beta(2),alphap(2),cp(2),rp(2),
     +           ca_2, c2a_2, clr_2, E)
      return
300   continue
      call O2_VV(R,nmax(3),alpha(3),beta(3),alphap(3),cp(3),rp(3),
     +           ca_3, c2a_3, clr_3, E)
      return
400   continue
      call O2_VV(R,nmax(4),alpha(4),beta(4),alphap(4),cp(4),rp(4),
     +           ca_4, c2a_4, clr_4, E)
      return
500   continue
      call O2_VV(R,nmax(5),alpha(5),beta(5),alphap(5),cp(5),rp(5),
     +           ca_5, c2a_5, clr_5, E)
      return
600   continue
      call O2_VV(R,nmax(6),alpha(6),beta(6),alphap(6),cp(6),rp(6),
     +           ca_6, c2a_6, clr_6, E)
      return
700   continue
      call O2_VV(R,nmax(7),alpha(7),beta(7),alphap(7),cp(7),rp(7),
     +           ca_7, c2a_7, clr_7, E)
      return
800   continue
      call O2_VV(R,nmax(8),alpha(8),beta(8),alphap(8),cp(8),rp(8),
     +           ca_8, c2a_8, clr_8, E)
      return

      end


      subroutine O2_VV (R, nm, a, b, ap, cp, rp, ca, c2a, clr, E)
      implicit none
      integer nm
      real*8 R, a, b, ap, cp, rp, ca(nm+1), c2a(nm+1), clr(4), E
c
c   Functional form of the potential energy curves 
c   (see also Eq. (15) in the article)
c
c       E (R, i) = 
c         sum_{n=0}^{nm} ca(n+1) * (R - 2.8)^n * exp[-a*(R - 2.8)]
c       + sum_{n=0}^{nm} c2a(n+1) * (R - 2.8)^n * exp[-2*a*(R - 2.8)]
c       + sum_{n=1}^{4} clr(n) * f_long (R, b, pow(n)) 
c
c   where pow(n) = 5, 6, 8, and 10 for n = 1, 2, 3, and 4 respectively
c   and f_long is a Tang-Toennies damped long range function. (See
c   routine f_long).
c
c   For R values smaller than the data point with the smallest R value
c   used in the fit (R < rp) an exponential extrapolation is used
c
c       E (R<rp, i) = cp * exp (-ap * (R - rp))
c
c
c
      integer n, pow(4)
      data pow/5,6,8,10/
      real*8 sum1, sum2, sum3, rs, rsn, f_long
c
      if (R.LT.rp) then
        E = cp*dexp(-ap*(R-Rp))
      else
        rs = R-2.8d0
        sum1 = 0.0d0
        sum2 = 0.0d0
        rsn = 1
        do n = 0, nm
          sum1 = sum1 + ca(n+1) * rsn
          sum2 = sum2 + c2a(n+1) * rsn
          rsn = rsn * rs
        end do
        sum3 = 0.0d0
        do n = 1, 4
          sum3 = sum3 + clr(n) * f_long(R,b,pow(n))
        end do
        E = sum1 * dexp(-a * rs) + 
     +      sum2 * dexp(-2.0d0 * a * rs) + sum3 
      end if
c
      end


      real*8 function f_long(r,beta,n)
      implicit none
      real*8 r, beta
      integer n
c
c     f_long(r,beta,n)=d_n(beta*r,n)/r^n
c
c     Tang-Toennies damping function:
c     d_n(x,n) = 1 - exp(-x) * sum_k=0:n x^k / k!
c
c     For small x (when d_n<1e-8) use:
c     d_n(x,n) = exp(-x)* sum_k=n+1:infty x^k/k!
c
      real*8  sum, term, dex, rn, dn, x
      integer i
c
      if (r.eq.0.0d0) then
         f_long=0.0d0
         return
      endif
c
      x=beta*r
      sum=1.0d0
      term=1.0d0
      rn=1.0d0
      do i=1,n
         rn=rn*r
         term=term*x/dfloat(i)
         sum=sum+term
      end do
      dex=dexp(-x)
      dn=1.0d0-dex*sum
      if (dabs(dn).lt.1.0d-8) then
c
c        alternative formula for small x
c
         sum=0.0d0
         do i=n+1,1000
            term=term*x/dfloat(i)
            sum=sum+term
            if (term/sum .lt. 1.0d-8) go to 10
         enddo
         write(*,*)'** ERROR in F_LONG: no convergence, x=',x,' n=',n
         stop 232
10       continue
         dn=sum*dex
      endif
      f_long=dn/rn
      end


      subroutine O2_NA (R, i, j, E)
      implicit none
      integer i, j
      real*8 R, E
c
      include 'O2_NA_data.h'
c
c     phase of NA coupling matrix element. 
c     phase = 0 -> < i | d/dR | j > = 0
c
      integer phase(8,8)
      data phase/
     +  0, 0, 0, 0, 0, 0, 0, 0,
     +  0, 0, 0, 0, 0, 0, 0, 0,
     +  0, 0, 0, 0, 0, 0, 0, 0,
     +  0, 0, 0, 0, 0, 0, 0, 0,
     +  0, 0, 0, 0, 0, 1, 0, 0,
     +  0, 0, 0, 0,-1, 0, 0, 0,
     +  0, 0, 0, 0, 0, 0, 0, 0,
     +  0, 0, 0, 0, 0, 0, 0, 0/

      if (i.lt.1 .or. i.gt.8) then
        write (*,*)  '** ERROR in O2_NA: i not between 1 and 8, i =', i
        stop 240
      else if (j.lt.1 .or. j.gt.8) then
        write (*,*)  '** ERROR in O2_NA: j not between 1 and 8, j =', j
        stop 241
      end if

      if (phase(i,j)) 100, 200, 300
100   call NA (R, c1, c2, c, alpha1, alpha2, r1, r2, E)
      E = -1.0d0 * E
      return
200   E = 0.0d0
      return
300   call NA (R, c1, c2, c, alpha1, alpha2, r1, r2, E)
      return
      end


      subroutine NA (R, c1, c2, c, a1, a2, r1, r2, E)
      implicit none
      real*8 R, c1, c2, c, a1, a2, r1, r2, E
c
c     Functional form of the non-adiabatic coupling matrix element
c       E = (c1 / (1 + c*(exp(-a1*(R-r1)) + exp(a1*(R-r1))))) +
c           (c2 / (1 + c*(exp(-a2*(R-r2)) + exp(a2*(R-r2)))))
c    
      real*8 Rs1, Rs2, fac1, fac2
c
      Rs1 = a1 * (R - r1)
      Rs2 = a2 * (R - r2)
      fac1 = c1 / (1.0d0 + c*(dexp(-Rs1) + dexp(Rs1)))
      fac2 = c2 / (1.0d0 + c*(dexp(-Rs2) + dexp(Rs2)))
      E = fac1 + fac2
c
      return
      end



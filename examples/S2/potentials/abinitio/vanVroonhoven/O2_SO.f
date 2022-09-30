      subroutine O2_SO (R, i, j, E)
      implicit none
      integer i, j
      real*8 R, E
c
      include 'O2_SO_data.h'
c
c     reduced matrix element <i || Hso || j> has number matelnr(i,j)
c
      integer matelnr(8,8)
      data matelnr/
     +   0,  0,  1,-13,  2,  3,  0,  0,
     +   0,  0,  0, 10, 17, 18,  0,  0,
     +   1,  0, 19,  4,  0,  0,  7,  0,
     +  13, 10, -4, 14, -5, -6, 15, 11,
     +   2,-17,  0,  5,  0,  0,  8,-20,
     +   3,-18,  0,  6,  0,  0,  9,-21,
     +   0,  0,  7,-15,  8,  9, 16,-12,
     +   0,  0,  0, 11, 20, 21, 12,  0/
c
c     conversion factors. Molpro calculates specific matrix elements
c     <(L) Lambda S Sigma ; R | Hso | (L') Lambda' S' Sigma'>
c     These are conversion factors from the specific matrix element
c     calculated by Molpro and the reduced matrix element
c     <(L) Lambda S ; R || Hso || (L') Lambda' S'>
c
      real*8 factors(21)
      data factors/
     +   -3.464101615137754d+00,   -2.449489742783178d+00,
     +   -2.449489742783178d+00,   -4.898979485566356d+00,
     +   -3.464101615137754d+00,   -3.464101615137754d+00,
     +   -1.095445115010332d+01,   -7.745966692414834d+00,
     +   -7.745966692414834d+00,   -2.449489742783178d+00,
     +   -7.745966692414834d+00,   -5.477225575051661d+00,
     +    1.732050807568877d+00,    2.449489742783178d+00,
     +   -3.162277660168380d+00,    5.477225575051661d+00,
     +   -1.732050807568877d+00,   -1.732050807568877d+00,
     +    2.449489742783178d+00,    3.162277660168380d+00,
     +    3.162277660168380d+00/
      integer mnr
      real*8 factor

      if (i.lt.1 .or. i.gt.8) then
        write (*,*)  '** ERROR in O2_SO: i not between 1 and 8, i =', i
        stop 240
      else if (j.lt.1 .or. j.gt.8) then
        write (*,*)  '** ERROR in O2_SO: j not between 1 and 8, j =', j
        stop 241
      end if

      mnr = abs(matelnr(i,j))
      if (mnr .gt. 0) then
        factor = factors(mnr) * dble(sign(1,matelnr(i,j)))
      end if

      goto (100, 200, 300, 400, 500, 600, 700, 800, 900,1000,
     +     1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100) mnr
c
c     mnr = 0
c
      E = 0.0d0
      return
 100  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_1, coefs_1, factor, E)
      return
 200  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_2, coefs_2, factor, E)
      return
 300  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_3, coefs_3, factor, E)
      return
 400  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_4, coefs_4, factor, E)
      return
 500  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_5, coefs_5, factor, E)
      return
 600  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_6, coefs_6, factor, E)
      return
 700  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_7, coefs_7, factor, E)
      return
 800  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_8, coefs_8, factor, E)
      return
 900  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_9, coefs_9, factor, E)
      return
1000  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_10, coefs_10, factor, E)
      return
1100  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_11, coefs_11, factor, E)
      return
1200  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_12, coefs_12, factor, E)
      return
1300  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_13, coefs_13, factor, E)
      return
1400  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_14, coefs_14, factor, E)
      return
1500  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_15, coefs_15, factor, E)
      return
1600  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_16, coefs_16, factor, E)
      return
1700  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_17, coefs_17, factor, E)
      return
1800  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_18, coefs_18, factor, E)
      return
1900  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_19, coefs_19, factor, E)
      return
2000  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_20, coefs_20, factor, E)
      return
2100  continue
      call SO (R, Aout(mnr), Bout(mnr), alpha(mnr), Ain(mnr), Bin(mnr),
     +         n_int(mnr), breaks_21, coefs_21, factor, E)
      return
      end


      subroutine SO (R, Aout, Bout, alpha, Ain, Bin, n_int, br, c, 
     +               factor,E)
      implicit none
      integer n_int
      real*8 R, Aout, Bout, alpha, Ain, Bin, br(n_int+1), c(4,n_int), 
     +       factor, E
c
c    Functional form of the reduced spin orbit matrix elements
c    for R < br(1)  
c       E = Ain + Bin * exp (2.5 (R - br(1)))
c    for R >= br(1) & R <= br(n_int+1)
c       E is represented by a piecewise polynomial. 
c       On interval i (which runs from br(i) to br(i+1)) this is 
c       defined by
c       E = (R - br(i))^3 * c(1,i) + (R - br(i))^2 * c(2,i)
c           + (R - br(i)) * c(3,i) + c(4,i)
c    for R > br(n_int+1)
c       E = Aout + Bout * exp (-alpha (R - br(n_int+1)))
c
c    
      integer i_int
      real*8 Rs
c
      if (R .lt. br(1)) then
        E = Ain + Bin * dexp(2.5d0 * (R - br(1)))
      else if (R .gt. br(n_int+1)) then
        E = Aout + Bout * dexp (-alpha * (R - br(n_int+1)))
      else
        i_int = 1
        do while (R .gt. br(i_int+1))
          i_int = i_int+1
        end do
        Rs = R - br(i_int)
        E = ((Rs**3) * c(1,i_int)) + ((Rs**2) * c(2,i_int)) +
     +      (Rs * c(3,i_int)) + c(4,i_int)
      end if
c     convert from wavenumber to atomic units, apply multiplication factor 
c     for conversion from specific matrix element to reduced matrix element
      E = E * factor / 2.1947463068d5
      return
      end




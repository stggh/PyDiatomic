========================= README.TXT ===============================

ARTICLE INFORMATION:   E-JCPSA6-116-303204           22 January, Issue # 4

Journal:    J. Chem. Phys. 116, XXXX (2002) 

Authors:   Mirjam C. G. N. van Vroonhoven and Gerrit C. Groenenboom

Title: Photodissociation of O_2 in the Herzberg continuum I: calculation of potential energy curves and properties

DEPOSIT INFORMATION

Description:
   Fortran 77 routines to evaluate the potential energy curves, reduced spin orbit matrix elements, and the non-adiabatic radial derivative coupling element. An example program with input and corresponding output is also provided.

Total No. of Files: 9

Filenames: README.TXT +      8 files               
           
           O2_V.f         routines to evaluate the potential curves

           O2_V_data.h    contains data statements to be included in O2_V.f

           O2_SO.f        routines to evaluate the spin orbit matrix elements

           O2_SO_data.h   contains data statements to be included in O2_SO.f

           O2_NA.f        routines to evaluate the non-adiabatic coupling matrix element

           O2_NA_data.h   contains data statements to be included in O2_SO.f  
     
      test.f         test program
            
      test.out       output for test program

Filetypes: all ASCII.
Special Instructions: See Below
Contact Information:
        Drs. M. C. G. N. van Vroonhoven
        Institute of Theoretical Chemistry
        Univ. of Nijmegen
        Toernooiveld 1
        6525 ED Nijmegen, THE NETHERLANDS
         Phone: +31-243653033  Fax:   +31-243653041  Email: mirjamv@theochem.kun.nl

Special instructions:

(1) Instructions for usage of the potential routines

(2) Instructions for usage of the spin orbit coupling routines

(3) Instructions for usage of the radial derivative coupling routines

======================================================
(1) Instructions for usage of the potential routines
======================================================

The file O2_V.f contains program test (misschien weg -> test programma apart) 
    subroutine O2_V
    subroutine O2_VV
    subroutine f_long

The calling sequence to evaluate one of the potentials is

      integer I
      
      real*8 R, E

      call O2_V (R, I, E)

where R is the internuclear distance coordinate, I is the sequence number of

the state (see below), and E is the resulting energy (R and I are input

parameters, E is the output parameter). The sequence numbers of the states are

defined in the table below, L, Lambda and S are the Hund's case (a) quantum

numbers of the states:
      -------------------------------------------
        I    name                L   Lambda  S     
      -------------------------------------------
        1    1  1Pi_u            1     1     0         
        
        2    c  1Sigma_u_minus   1     0     0       

        3    A' 3Delta_u         2     2     1        

        4    1  3Pi_u            2     1     1         

        5    A  3Sigma_u_plus    0     0     1         

        6    2  3Sigma_u_plus    2     0     1                 

        7    1  5Pi_u            1     1     2         

        8    1  5Sigma_u         1     0     2          
      -------------------------------------------               

The routines O2_VV and f_long are called by the routine O2_V in the

evaluation. To compile the routine O2_V, the file O2_V_data.inc is needed,

since it contains data statements to be included in the routine O2_V.

==============================================================

(2) Instructions for usage of the spin orbit coupling routines

==============================================================

The file O2_SO.f contains

      program test (misschien weg -> apart testprogramma)

      subroutine O2_SO

      subroutine SO

The calling sequence to evaluate the reduced matrix element

< I || Hso (R) || J > is

      integer I, J

      real*8 R, E

      call O2_SO (R, I, J, E)

where R is the internuclear distance, I is the sequence number of the bra

state, J is the sequence number of the ket state, and E is the resulting

reduced matrix element (Input parameters: R, I, and J; output parameter: E).

The state sequence numbers are defined in the table above (see "(1)

Instructions for usage of the potential routines"). This table defines only

states with Lambda >= 0. From Eq. (10) of our paper can be derived that

  < (L) -Lambda S || Hso || (L') -Lambda' S' > =
  
    (-1)^(L + L' + 1) < (L) Lambda S || Hso || (L') Lambda' S' >

which defines the reduced matrix elements for states with Lambda < 0.

The routine O2_SO includes (at compile time) data statements from the file

O2_SO_data.inc. 


=====================================================================

(3) Instructions for usage of the radial derivative coupling routines

=====================================================================

The file O2_NA.f contains

      program test (misschien weg -> apart testprogramma)

      subroutine O2_NA

      subroutine NA

The calling sequence to evaluate the radial derivative coupling

< I | d / d R | J > is

      integer I, J

      real*8 R, E

      call O2_NA (R, I, J, E)

where R is the internuclear distance, I and J are the sequence number of the bra and ket states respectively, and E is the resulting non-adiabatic coupling matrix element (Input parameters: R, I, and J; output parameter: E). E will
only be non-zero when I = 5 and J = 6, or when I = 6 and J = 5, since there is only a radial derivative coupling between A and 2 3Sigma_ungerade_plus, state numbers 5 and 6.  The routine O2_NA includes datastatements from the file

O2_NA_data.inc (only needed at compile time), and calls the routine NA for the actual calculation.

=====================================================================

Nijmegen, July 1, 2001

=====================================================================


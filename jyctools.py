"""
JYC.PY
--------------------------------------------------------------------------------
Class definining "Admissibility" via outer and inner approximation schemes.
Prepared for ECON8001 Topics in Economic Dynamics taught at the ANU School 
of Economics.

Notes:
--------------------------------------------------------------------------------
   See Judd, Yeltekin and Conklin (2003, Econometrica). 
   Requires CVXOPT and GLPK   interface, and, MPI4PY. 
   CVXOPT must be custom built as per the instructions on 
   cvxopt.org to interface with GLPK.
________________________________________________________________________________
      (c) 2013--, T. Kam. Email: tcy.kam@gmail.com

CHANGELOG: Implemented Algorithm 3 (inner approx) from JYC. 
Previous version of admit_inner() was a cruder discretized version.
"""
import numpy as np
import scipy.spatial as SpySpatial
from cvxopt import glpk, matrix, solvers
from mpi4py import MPI
from matplotlib import pyplot as plt
from matplotlib import animation, cm
import itertools

class jyc:

    def __init__(self, DELTA, u_vecm, u_vecm_deviate, H):
        """
        JYC.PY: Polytope algorithm for supergames: JYC method
        (c) 2015, Timothy Kam (tcy.kam@gmail.com)
        """
        self.DELTA = DELTA
        self.u_vecm = u_vecm
        self.u_vecm_deviate = u_vecm_deviate
        self.H = H

    def admit_outer(self, l_idx, W_old, c_old):
        """
        Algorithm 1 in JYC (outer approximation of the APS monotone convex set-valued operator): Admissibility.
        """
        N_player, N_profile = self.u_vecm.shape
        h_l = self.H[l_idx,:]    # Current spherical code, direction l
        w_set = np.zeros((N_player,N_profile)) # Preallocate
        cplus = np.zeros((N_profile,1))
        w_min = np.amin(W_old, axis=0)
        w_max = np.amax(W_old, axis=0)

        # Convert to CVXOPT dtype='d' matrix (vector defaults to column vector)
        H_set = matrix(self.H)
        c_old_set = matrix(c_old)
        F_set_ub = matrix(np.eye(N_player))        # Feasible hypercube: GLPK
        F_set_lb = matrix(np.eye(N_player))
        w_ub = matrix(w_max)
        w_lb = matrix(w_min)

        # Loop over all action profile
        for a_idx in range(N_profile):

            # Weighted value from action-promise pair (a,w)
            u_now = (1-self.DELTA)*self.u_vecm[:,a_idx] # Current payoff
            current_payoff = h_l.dot(u_now)             # weighted
            continue_payoff = h_l*self.DELTA            # weighted

            # Max-min value from current deviation to (a',w_min)
            v_maxmin = (1.-self.DELTA)*self.u_vecm_deviate[:,a_idx] \
                                                    + self.DELTA*w_min
            # Deviation payoff gain
            v_diff = matrix(u_now - v_maxmin)

            # LP problem over promises w
            IC_set = matrix([[-self.DELTA,      0.       ], \
                             [0.,           -self.DELTA  ]])

            A = matrix([ H_set,     \
                         IC_set,    \
                         -F_set_lb,  \
                         F_set_ub   ])

            b = matrix([ c_old_set, \
                         v_diff,    \
                         -w_lb,      \
                         w_ub       ])

            # Linear objective function: continue_payoff
            c = matrix( continue_payoff )

            # Put to GLPK!
            glpk.options['msg_lev']='GLP_MSG_OFF'
            sol=glpk.lp(-c,A,b)
            if (sol[0] == 'optimal'):
                w_opt = np.array(sol[1]).reshape(N_player)	# optimizers
                cplus[a_idx] = current_payoff + np.dot(c, w_opt)
                w_set[:,a_idx] = w_opt # Store optimizers
                exitflag = 0
            else:
                cplus[a_idx] = -1e5
                exitflag = -1
        return cplus, w_set, exitflag

    def aps_outer(self, W_old, c_old):
        """Define the JYC outer approximation of the APS monotone convex set-valued operator"""
        L, N_player = self.H.shape
        # MPI4PY parallelization: START
        # =====================================================================
        # ROOT 0 WORLD ...
        COMM = MPI.COMM_WORLD         # instantiate MPI worlds
        num_cpus = COMM.size
        rank = COMM.rank
        # Shape of partition elements of root=0 domain
        nr = int(np.ceil(float(L)/num_cpus)*num_cpus) # No. rows
        nc = 1                                        # No. columns
        # Domain partition element (slave's slice of job)
        block_size = nr / num_cpus
        L_partition = np.zeros((block_size,nc),dtype='int')
        # CODOMAINs: Root 0's Gather destination(s) at the end
        c_new = np.zeros((nr,nc), dtype='d')
        W_new = np.zeros((nr,N_player), dtype='d')

        # Setup: Broadcast Arrays to workers
        if (rank == 0):
            BcastData = {   'self': self,    \
                            'W_old': W_old,  \
                            'c_old': c_old
                            }
            L_set = np.arange(L, dtype='int')
        else:
            BcastData = None
            L_set = np.zeros((L,nc), dtype='int')

        # Scatter L_set to slaves each with L_partition
        COMM.Scatter([L_set,MPI.INT],[L_partition,MPI.INT])

        # ---------- LOCAL SLAVE WORLD: start---------------------------------#
        # Time counter: start
        #wt0 = MPI.Wtime()
        # Broadcast to all workers
        BcastData = COMM.bcast( BcastData, root=0 )
        self_local = BcastData['self']
        W_old_local = BcastData['W_old']
        c_old_local = BcastData['c_old']
        # At each worker level: Do subset of repeated GLPK jobs at each (l,a)
        c_new_local = np.zeros(L_partition.shape)
        W_new_local = np.zeros((len(c_new_local),2))
        for i, l_idx in enumerate(L_partition):
            cplus,w_set,exit = \
                        self_local.admit_outer(l_idx,W_old_local,c_old_local)
            c_new_local[i] = cplus.max()        # Max over all a's
            amax_idx = cplus.argmax()           # Boolean index of optimal a
            W_new_local[i,:] = (1-self_local.DELTA) \
                                    *self_local.u_vecm[:,amax_idx] \
                                    + self_local.DELTA*w_set[:,amax_idx]
                                    # Payoff coordinate in direction l_idx
        # Wall end time at local worker
        #wt1 = MPI.Wtime() - wt0
        #print("\nRank %d  ... \
        #      	\tMPI Wall Time = %6.8g seconds") % (rank, wt1)
        # ---------- LOCAL SLAVE WORLD: finish -------------------------------#

        # ---------- BACK TO MOTHERSHIP ROOT 0 again -------------------------#
        # Collect back to CODOMAINs in rank 0
        COMM.Gather([c_new_local,MPI.DOUBLE],[c_new,MPI.DOUBLE])
        COMM.Gather([W_new_local,MPI.DOUBLE],[W_new,MPI.DOUBLE])

        # Take MAX of all worker wall times
        #TotalTime = COMM.reduce(wt1, op=MPI.MAX, root=0)
        #if (rank==0):
        #    print "\nRank %d says: ''Total Time = %6.8g seconds''.\n" \
        #                                                % (rank, TotalTime)
        return W_new, c_new

###============================================================================
    def admit_inner(self, l_idx, ConvexHull_W_old):
        """Algorithm 3 (Step 1(a)) (Inner Monotone Hyperplane Approx.) in JYC's paper. Uses SCIPY.SPATIAL's ConvexHull implementation of the QHULL software."""

        h_l = self.H[l_idx,:]              # Current spherical code, direction l
        N_player, N_profile = self.u_vecm.shape
        Z_set = ConvexHull_W_old.points        # Original points in W
        vert_ind = ConvexHull_W_old.vertices # Extreme points/vertices
        facet_eqn = ConvexHull_W_old.equations # Linear Inequalities for facets

        # Worst values--attained at a vertex since co(W) is rep. by polytope
        w_min = np.amin(Z_set[vert_ind,:], axis=0)
        w_max = np.amax(Z_set[vert_ind,:], axis=0)

        w_temp = np.zeros((N_player,N_profile))#np.tile(-np.inf, (N_player,N_profile)) # Preallocate
        c_temp = np.zeros((N_profile,1)) #np.tile(-np.inf, (N_profile,1))#np.zeros((N_profile,1))
        # Convert to CVXOPT dtype='d' matrix (vector defaults to column vector)
        G_set = matrix(facet_eqn[:,0:N_player])   # Normals to facets of co(W)
        #print np.sum(np.sum(G_set**2, axis=1))
        c_set = matrix(-facet_eqn[:,-1])           # Levels of facets of co(W)
        #print facet_eqn[:,-1].max(), facet_eqn[:,-1].min()
        F_set_ub = matrix(np.eye(N_player))       # Feasible hypercubes: GLPK
        F_set_lb = matrix(np.eye(N_player))
        w_ub = matrix(w_max)
        w_lb = matrix(w_min)
        for a_idx in range(N_profile):
            # Weighted value from action-promise pair (a,w)
            u_now = (1-self.DELTA)*self.u_vecm[:,a_idx] # Current payoff
            current_payoff = h_l.dot(u_now)             # weighted
            continue_payoff = h_l*self.DELTA            # weighted

            # Max-min value from current deviation to (a',w_min)
            v_maxmin = (1.-self.DELTA)*self.u_vecm_deviate[:,a_idx] \
                                                    + self.DELTA*w_min
            # Deviation payoff gain
            v_diff = matrix(u_now - v_maxmin)

            # LP problem over promises w
            IC_set = matrix([[-self.DELTA,      0.       ], \
                             [0.,           -self.DELTA  ]])

            A = matrix([ G_set,     \
                         IC_set,    \
                         -F_set_lb,  \
                         F_set_ub   ])

            b = matrix([ c_set, \
                         v_diff,    \
                         -w_lb,      \
                         w_ub       ])

            # Linear objective function: continue_payoff
            c = matrix( continue_payoff )

            # Put to GLPK!
            glpk.options['msg_lev']='GLP_MSG_OFF'
            sol=glpk.lp(-c,A,b)
            if (sol[0] == 'optimal'):
                w_opt = np.array(sol[1]).reshape(N_player)	# optimizers
                c_temp[a_idx] = current_payoff + np.dot(c, w_opt)
                w_temp[:,a_idx] = w_opt # Store optimizers
                exitflag = 0
            else:
                c_temp[a_idx] = -np.inf
                exitflag = -1
        return c_temp, w_temp, exitflag

    def aps_inner(self, Z_old):
        """Algorithm 3, Step 1(b-c) and Step 2 of JYC. Define the JYC inner approximation of the APS monotone convex set-valued operator. See also ADMIT_INNER"""
        L, N_player = self.H.shape
        # MPI4PY parallelization: START
        # =====================================================================
        # ROOT 0 WORLD ...
        COMM = MPI.COMM_WORLD         # instantiate MPI worlds
        num_cpus = COMM.size
        rank = COMM.rank
        # Shape of partition elements of root=0 domain
        nr = int(np.ceil(float(L)/num_cpus)*num_cpus) # No. rows
        nc = 1                                        # No. columns
        # Domain partition element (slave's slice of job)
        block_size = nr / num_cpus
        L_partition = np.zeros((block_size,nc),dtype='int')
        # CODOMAINs: Root 0's Gather destination(s) at the end
        #c_new = np.zeros((nr,nc), dtype='d')
        Z_new = np.zeros((nr,N_player), dtype='d')
        # Take convex hull of W_old

        # Setup: Broadcast Arrays to workers
        if (rank == 0):
            Whull = SpySpatial.ConvexHull(Z_old)
            BcastData = {   'self': self,    \
                            'Whull': Whull
                        }
            L_set = np.arange(L, dtype='int')
        else:
            BcastData = None
            L_set = np.zeros((L,nc), dtype='int')

        # Scatter L_set to slaves each with L_partition
        COMM.Scatter([L_set,MPI.INT],[L_partition,MPI.INT])

        # ---------- LOCAL SLAVE WORLD: start---------------------------------#
        # Time counter: start
        #wt0 = MPI.Wtime()
        # Broadcast to all workers
        BcastData = COMM.bcast( BcastData, root=0 )
        self_local = BcastData['self']
        Whull_local = BcastData['Whull']
        #    c_old_local = BcastData['c_old']
        # At each worker level: Do subset of repeated GLPK jobs at each (l,a)
        #c_new_local = np.zeros(L_partition.shape)
        Z_new_local = np.zeros((len(L_partition),2))
        for i, l_idx in enumerate(L_partition):
            c_temp,w_temp,exit = \
                        self_local.admit_inner(l_idx,Whull_local)
            #c_new_local[i] = c_temp.max()        # Max over all a's
            amax_idx = c_temp.argmax()           # Boolean index of optimal a
            Z_new_local[i,:] = (1-self_local.DELTA) \
                                    *self_local.u_vecm[:,amax_idx] \
                                    + self_local.DELTA*w_temp[:,amax_idx]
                                    # Payoff coordinate in direction l_idx
            #print Z_new_local[i,:]
        # Wall end time at local worker
        #wt1 = MPI.Wtime() - wt0
        #print("\nRank %d  ... \
        #      	\tMPI Wall Time = %6.8g seconds") % (rank, wt1)
        # ---------- LOCAL SLAVE WORLD: finish -------------------------------#

        # ---------- BACK TO MOTHERSHIP ROOT 0 again -------------------------#
        # Collect back to CODOMAINs in rank 0
        #COMM.Gather([c_new_local,MPI.DOUBLE],[c_new,MPI.DOUBLE])
        COMM.Gather([Z_new_local,MPI.DOUBLE],[Z_new,MPI.DOUBLE])

        # Take MAX of all worker wall times
        #TotalTime = COMM.reduce(wt1, op=MPI.MAX, root=0)
        #if (rank==0):
        #    print "\nRank %d says: ''Total Time = %6.8g seconds''.\n" \
        #                                                % (rank, TotalTime)
        #return Z_new, c_new
        return Z_new

###============================================================================

    def admit_inner_discrete(self, l_idx, ConvexHull_W_old, M):
        """
        JYC inner approximation of the APS monotone convex set-valued operator: Admissibility. Uses a cruder discretized version of finding the normal vectors to the points on the facets (:=: face of 2D polygons) representing the inner approximant set B(W).
        Based on a MATLAB code by Pablo d'Erasmo on Dean Corbae's site.
        """
        vert = ConvexHull_W_old.points[ConvexHull_W_old.vertices,:]
        K = len(ConvexHull_W_old.vertices)
        #print K
        h_l = self.H[l_idx,:]    # Current spherical code, direction l
        N_player, N_profile = self.u_vecm.shape
        N_K = (K-1)*M
        # Storage:
        c_temp = np.zeros((N_K,1))
        W_temp = np.zeros((N_player,N_K))
        errflag_temp = np.zeros((N_K,1),dtype='int')

        # Storage for arrays exiting function
        W_la = np.zeros((N_player,N_profile))
        cplus = np.zeros((N_profile, 1))

        # Variable Weight

        weight = np.linspace(0.,1.,M)

        # Worst values
        w_min = np.amin(ConvexHull_W_old.points, axis=0)

        for a_idx in range(N_profile):
            j = 0
            for n in range(K-1):
                for m in range(M):
                    # Find value from pair (a,w) where w is a weighted average of two vertices in VERT
                    lam = weight[m]
                    w = (1.-lam)*vert[n,:] + lam*vert[n+1,:]  #(n,m)-th guess
                    # (a,w) induces payoff differential
                    v_enforce = (1.-self.DELTA)*self.u_vecm[:,a_idx]   \
                                                                + self.DELTA*w
                    v_deviate = (1.-self.DELTA)*self.u_vecm_deviate[:,a_idx] \
                                                            + self.DELTA*w_min
                    IC = v_enforce - v_deviate          # > 0 for admissibility
                    #print j
                    #print ((IC[IC >=0]).size == IC.size)
                    if ((IC[IC >=0]).size == IC.size):
                        #print np.dot(h_l, v_enforce)
                        c_temp[j] = np.dot(h_l, v_enforce)
                        #print v_enforce
                        W_temp[:,j] = v_enforce
                        errflag_temp[j] = 1
                    else:
                        c_temp[j] = -np.inf
                        errflag_temp[j] = -1
                    j += 1
                #print j

            # Update levels and payoff vectors at each action a_idx
            cplus[a_idx] = c_temp.max()
            n_max_idx = c_temp.argmax()
            W_la[:,a_idx] = W_temp[:,n_max_idx]
            exitflag = errflag_temp[n_max_idx]
            # Override level if Admissibility not satisfied at A[a_idx,:]
            if (exitflag == -1):
                cplus[a_idx] = -np.inf
        return W_la, cplus, exitflag

    def aps_inner_discrete(self, W_old, c_old, M):
        """Define the JYC inner approximation of the APS monotone convex set-valued operator. Uses a cruder discretized version of finding the normal vectors to the points on the facets (:=: face of 2D polygons) representing the inner approximant set B(W).
        (Based on a MATLAB code by Pablo d'Erasmo on Dean Corbae's site.)
        """
        L, N_player = self.H.shape
        #c_new = c_old.copy()
        #W_new = W_old.copy()

        # MPI4PY parallelization: START
        # =====================================================================
        # ROOT 0 WORLD ...
        COMM = MPI.COMM_WORLD         # instantiate MPI worlds
        num_cpus = COMM.size
        rank = COMM.rank
        # Shape of partition elements of root=0 domain
        nr = int(np.ceil(float(L)/num_cpus)*num_cpus) # No. rows
        nc = 1                                        # No. columns
        # Domain partition element (slave's slice of job)
        block_size = nr / num_cpus
        L_partition = np.zeros((block_size,nc),dtype='int')
        # CODOMAINs: Root 0's Gather destination(s) at the end
        c_new = np.zeros((nr,nc), dtype='d')
        W_new = np.zeros((nr,N_player), dtype='d')

        # Setup: Broadcast Arrays to workers
        if (rank == 0):
            Whull = SpySpatial.ConvexHull(W_old)
            BcastData = { 'self': self, \
                          'Whull': Whull, \
                          'M': M
                        }
            L_set = np.arange(L, dtype='int')
        else:
            BcastData = None
            L_set = np.zeros((L,nc), dtype='int')

        # Scatter L_set to slaves each with L_partition
        COMM.Scatter([L_set,MPI.INT],[L_partition,MPI.INT])

        # ---------- LOCAL SLAVE WORLD: start---------------------------------#
        # Time counter: start
        #wt0 = MPI.Wtime()
        # Extract broadcasts at slave local
        BcastData = COMM.bcast( BcastData, root=0 )
        self_local = BcastData['self']
        Whull_local = BcastData['Whull']
        M_local = BcastData['M']
        # At each worker level: Do subset of repeated GLPK jobs at each (l,a)
        c_new_local = np.zeros(L_partition.shape)
        W_new_local = np.zeros((len(c_new_local),2))
        for i, l_idx in enumerate(L_partition):
            W_la, cplus, exitflag = self_local.admit_inner(l_idx, \
                                                    Whull_local, M_local)
            # Update levels of hyperplane levels for inner
            c_new_local[i] = cplus.max()
            max_a_idx = cplus.argmax()
            # Derive corresponding payoff coordinates at hypoerplane normals
            W_new_local[i,:] = W_la[:,max_a_idx]
        # Wall end time at local worker
        #wt1 = MPI.Wtime() - wt0
        #print("\nRank %d  ... \
        #      	\tMPI Wall Time = %6.8g seconds") % (rank, wt1)
        # ---------- LOCAL SLAVE WORLD: finish -------------------------------#

        # ---------- BACK TO MOTHERSHIP ROOT 0 again -------------------------#
        # Collect back to CODOMAINs in rank 0
        COMM.Gather([c_new_local,MPI.DOUBLE],[c_new,MPI.DOUBLE])
        COMM.Gather([W_new_local,MPI.DOUBLE],[W_new,MPI.DOUBLE])
        # Take MAX of all worker wall times
        #TotalTime = COMM.reduce(wt1, op=MPI.MAX, root=0)
        #if (rank==0):
        #    print "\nRank %d says: ''Total Time = %6.8g seconds''.\n" \
        #                                                % (rank, TotalTime)
        return W_new, c_new

    def ubound_solve(self, c_new):
        """ For N = 2 only. Given H (L x N shape) and c_new (L x 1 shape), solve for x, such that H(l,:)*x(:l) = c(l:l+N-1), for all l = 0,...,L-1.
        """
        L, N_player = self.H.shape
        U = np.zeros((L,N_player))
        for l_idx in range(L):
            if (l_idx < L-1):
                l_idx_plus = l_idx + 1
            else:
                l_idx_plus = 0
            x = np.vstack((self.H[l_idx,:], self.H[l_idx_plus,:]))
            y = np.vstack((c_new[l_idx], c_new[l_idx_plus]))
            U[l_idx,:] = np.dot( np.linalg.pinv(x), y).T
        return U

    def Array_Distance(self, Matrix1, Matrix2):
        """ Calculate distance between ND-arrays """
        if (Matrix1.shape != Matrix2.shape):
            raise ValueError('Matrices should be of same shape!')
        else:
            Mdistance = (np.absolute(Matrix1 - Matrix2)).max()
        return Mdistance

    def Hasdorff_Distance_looped(self,Z1, Z2):
        """ Calculate distance between ND-arrays Z1, Z2 with different len(Z) """
        M2 = len(Z2)
        M1, N_player = Z1.shape
        d = np.zeros((M2,M1))
        for n in range(N_player):
            x2 = np.tile(Z2[:,n],(M1,1))
            x1 = np.tile(Z1[:,n],(M2,1))
            d = d + (x2.T - x1)**2.0
        d = np.sqrt(d)
        d1 = (np.min(d,axis=1)).max()
        d2 = (np.min(d,axis=0)).max()
        return np.max([d1,d2])

    def Hausdorff_Distance(self,Z1, Z2):
        """ Calculate distance between ND-arrays Z1, Z2 with different len(Z). Same as VERTICES_DISTANCE() but uses SCIPY.SPATIAL.DISTANCE """
        d = SpySpatial.distance.cdist(Z1,Z2, 'euclidean')
        h1 = np.max(np.min(d,axis=1))
        h2 = np.max(np.min(d,axis=0))
        return np.max([h1,h2])

    def Animate_Patches(self, V_seq, AnimateOptions):
        """
        Usage: V_seq is a List, where V_seq[i] is a N x 2 Numpy array
               AnimateOptions is a Dictionary
        """
        fig = plt.figure()
        if ('Resolution' in AnimateOptions):
            fig.set_dpi(AnimateOptions['Resolution'])
        else:
            fig.set_dpi(100)
        if ('Transparency' in AnimateOptions):
            transparency = AnimateOptions['Transparency']
        else:
            transparency = 0.5
        if ('ColorMap' in AnimateOptions):
            colormap = AnimateOptions['ColorMap']
        else:
            colormap = cm.hsv

        # Flatten the V_seq list of Numpy arrays to get max/min elements overall
        V, l1, l2 = self.flatten2(V_seq)
        xmin = ymin = 1.05*min(V)
        xmax = ymax = 1.05*max(V)
        ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))
        #colormap = cm.hsv
        step = 40
        def animate(i):
            patch = plt.Polygon(V_seq[i],                   \
                                fc=colormap(i / float(step)), \
                                alpha=transparency)
            ax.add_patch(patch)
            return patch

        anime = animation.FuncAnimation(fig, animate, frames=len(V_seq), \
                                                                interval=300)
        return anime

    # --------------- TOOLS SECTION ------------------------------------------#
    def flatten2(self, nl):
        """
        To flatten Python List of lists / numpy arrays (2 levels). (See also reverse operation in RECONSTRUCT() below.)
        Usage: L_flat,l1,l2 = flatten2(L)
        Source: http://stackoverflow.com/questions/27982432/flattening-and-unflattening-a-nested-list-of-numpy-arrays
        """
        l1 = [len(s) for s in itertools.chain.from_iterable(nl)]
        l2 = [len(s) for s in nl]

        nl = list(itertools.chain.from_iterable(itertools.chain.from_iterable(nl)))

        return nl,l1,l2

    def reconstruct2(self, nl, l1, l2):
        """
        To reconstruct Python List of lists / numpy arrays. Inverse operation of FLATTEN() above.
        Usage: L_reconstructed = reconstruct2(L_flat,l1,l2)
        Source: http://stackoverflow.com/questions/27982432/flattening-and-unflattening-a-nested-list-of-numpy-arrays
        """
        return np.split(np.split(nl,np.cumsum(l1)),np.cumsum(l2))[:-1]

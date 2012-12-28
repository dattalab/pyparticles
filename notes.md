# TODO #
* initial position diversity
    - get rid of particle factory, just build particles outside

* ipyton parallel
* synthetic 19D testing
* profile
* hdp concentration GS on smart particles

* GPU implementation of naive momentum particles

* flipper bit in particle

* Gibbs messages in pyhsmm!
    - two variants of VAR: standard and latent prefixes on switch
    - for standard, just give obs_distns a ref to data and let pyhsmm pass in
      indices, then slicify them

# Thinking #
* Options to try
    1. smart particles
    2. naive particles + Gibbs
    3. naive particles at start, then Gibbs, then smart particles
* kinematics: FK or IK


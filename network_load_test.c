/*
 * The entirety of this work is licensed under the Apache License,
 * Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License.
 *
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <network_test.h>
#include <ctype.h>
#include <unistd.h>
#include <stdbool.h>

#define VERSION 1.3

//#define NUM_NETWORK_TESTS 3
//#define NUM_CONGESTOR_TESTS 4
int NUM_NETWORK_TESTS = 0;
int NUM_CONGESTOR_TESTS = 0;
uint32_t NETWORK_TESTS = 0;
uint32_t CONGESTOR_TESTS = 0;
#define TB(x) (1 << (x)) /* test bit */
bool skip_baseline = false;

/* loop counts for the various tests */
#define NUM_LAT_TESTS 10000
#define NUM_LAT_RANDS 30
#define NUM_LAT_ITERS 200
#define NUM_BW_TESTS 10000
#define NUM_BW_RANDS 30
#define NUM_BW_ITERS 8
#define NUM_ALLREDUCE_TESTS 100000
#define NUM_ALLREDUCE_ITERS 200

/* test specific tuning */
#define BW_MSG_COUNT 16384
#define BW_OUTSTANDING 8

/* define where baseline sizes for congestor tests are */
#define A2A_BASE_NODES 32
#define INCAST_BASE_NODES 32
#define BCAST_BASE_NODES 32
#define ALLREDUCE_BASE_NODES 32

/* tuning for congestor tests */
#define A2A_MSG_COUNT 512
#define A2A_TESTS 256
#define A2A_BASELINE_ITERS 512
#define ALLREDUCE_MSG_COUNT 819200
#define ALLREDUCE_TESTS 256
#define ALLREDUCE_BASELINE_ITERS 512
#define INCAST_MSG_COUNT 512
#define INCAST_TESTS 256
#define INCAST_BASELINE_ITERS 512
#define BCAST_MSG_COUNT 512
#define BCAST_TESTS 256
#define BCAST_BASELINE_ITERS 512

#define CONGESTOR_NODE_FRAC 0.8

//CommTest_t network_tests_list[NUM_NETWORK_TESTS], congestor_tests_list[NUM_CONGESTOR_TESTS];
CommTest_t *network_tests_list;
CommTest_t *congestor_tests_list;

typedef struct CongestorResults_st {
     double *baseline_perf_hires;
     double *perf_hires;
     double baseline_perf;
     double perf;
} CongestorResults_t;

int network_test_setup(CommTest_t req_test, int *ntests, int *nrands, int *niters, char *tname, char *tunits)
{
     int nl = 64;

     switch (req_test) {
     case P2P_LATENCY:
          *ntests = NUM_LAT_TESTS;
          *nrands = NUM_LAT_RANDS;
          *niters = NUM_LAT_ITERS;
          snprintf(tname, nl, "%s (8 B)", "RR Two-sided Lat");
          snprintf(tunits, nl, "%s", "usec");
          break;
     case RMA_LATENCY:
          *ntests = NUM_LAT_TESTS;
          *nrands = NUM_LAT_RANDS;
          *niters = NUM_LAT_ITERS;
          snprintf(tname, nl, "%s (8 B)", "RR Get Lat");
          snprintf(tunits, nl, "%s", "usec");
          break;
     case P2P_BANDWIDTH:
          *ntests = NUM_BW_TESTS;
          *nrands = NUM_BW_RANDS;
          *niters = NUM_BW_ITERS;
          snprintf(tname, nl, "%s (%4i B)", "RR Two-sided BW", (int)(sizeof(double)*BW_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case RMA_BANDWIDTH:
          *ntests = NUM_BW_TESTS;
          *nrands = NUM_BW_RANDS;
          *niters = NUM_BW_ITERS;
          snprintf(tname, nl, "%s (%4i B)", "RR Put BW", (int)(sizeof(double)*BW_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case P2P_NEIGHBORS:
          *ntests = NUM_BW_TESTS;
          *nrands = NUM_BW_RANDS;
          *niters = NUM_BW_ITERS;
          snprintf(tname, nl, "%s (%4i B)", "RR Two-sided BW+Sync", (int)(sizeof(double)*BW_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case ALLREDUCE_LATENCY:
          *ntests = NUM_ALLREDUCE_TESTS;
          *nrands = 1;
          *niters = NUM_ALLREDUCE_ITERS;
          snprintf(tname, nl, "%s (8 B)", "Multiple Allreduce");
          snprintf(tunits, nl, "%s", "usec");
          break;
     default:
          break;
     }

     return 0;
}

int run_network_tests(CommConfig_t *config, int ncong_tests, MPI_Comm test_comm,
                      MPI_Comm local_comm, MPI_Comm global_comm)
{
     int itest = 0, ctest, myrank, niters, ntests, nrands;
     CommResults_t *results_base, *results, nullrslt;
     MPI_Request req;
     int nl=64;
     char tname[nl], tunits[nl];

     mpi_error(MPI_Comm_rank(local_comm, &myrank));

     results_base = malloc(sizeof(CommResults_t) * NUM_NETWORK_TESTS);
     results      = malloc(sizeof(CommResults_t) * NUM_NETWORK_TESTS);
     if (results_base == NULL || results == NULL) {
          die("Failed to allocate results in run_network_tests()\n");
     }

/*
 * Run the network baseline tests when they aren't skipped via the -s flag
 * or there are not congestor tests specified.
 */
if (!skip_baseline || !ncong_tests) {
     /* gather the baseline performance */
     print_header(config, 0, 0);
     for (itest = 0; itest < NUM_NETWORK_TESTS; itest++) {

          results_base[itest].distribution = NULL;
          network_test_setup(network_tests_list[itest], &ntests, &nrands, &niters, tname, tunits);
          if (network_tests_list[itest] != ALLREDUCE_LATENCY) {
               random_ring(config, 0, ntests, nrands, niters, network_tests_list[itest],
                           TEST_NULL, test_comm, local_comm, &results_base[itest]);
          } else {
               allreduce_test(config, ntests, niters, test_comm, local_comm, &results_base[itest]);
          }

          int from_min = 0;
          if (network_tests_list[itest] == P2P_LATENCY || network_tests_list[itest] == RMA_LATENCY ||
              network_tests_list[itest] == ALLREDUCE_LATENCY) from_min = 1;
          print_results(config, myrank, 1, from_min, tname, tunits, &results_base[itest]);

          if (myrank == 0) {
               write_distribution(network_tests_list[itest], TEST_NULL, 1, &results_base[itest], tname, tunits);
          }

     }
}

     /* barrier to prevent congestors from communicating while we get baselines */
     mpi_error(MPI_Barrier(global_comm));

/* Bail now if there are no congestion tests. Network baseline only. */
if (!ncong_tests)
     return 0;

     /* allow congestor tests to get their baselines */
#ifdef VERBOSE
if (!skip_baseline) {
     print_header(config, 1, 0);
     for (ctest = 0; ctest < ncong_tests; ctest++) {
          print_results(config, myrank, 0, 0, " ", " ", &nullrslt);
     }
}
#endif
     mpi_error(MPI_Barrier(global_comm));

     /* now loop over tests where we run congestor tests at the same time */
#ifndef VERBOSE
     print_header(config, 2, network_tests_list[itest]);
#endif
     for (itest = 0; itest < NUM_NETWORK_TESTS; itest++) {

          results[itest].distribution = NULL;
#if VERBOSE
          print_header(config, 2, network_tests_list[itest]);
#endif

          /* allow the congestors to get started and then release us once they are loading the network */
          mpi_error(MPI_Ibarrier(global_comm, &req));
          mpi_error(MPI_Wait(&req, MPI_STATUS_IGNORE));

          network_test_setup(network_tests_list[itest], &ntests, &nrands, &niters, tname, tunits);
          if (network_tests_list[itest] != ALLREDUCE_LATENCY) {
               random_ring(config, 0, ntests, nrands, niters, network_tests_list[itest],
                           TEST_CONGESTORS, test_comm, local_comm, &results[itest]);
          } else {
               allreduce_test(config, ntests, niters, test_comm, local_comm, &results[itest]);
          }

          /* now let the congestor tests know we are done */
          mpi_error(MPI_Ibarrier(global_comm, &req));
          mpi_error(MPI_Wait(&req, MPI_STATUS_IGNORE));

          int from_min = 0;
          if (network_tests_list[itest] == P2P_LATENCY || network_tests_list[itest] == RMA_LATENCY ||
              network_tests_list[itest] == ALLREDUCE_LATENCY) from_min = 1;
          print_results(config, myrank, 1, from_min, tname, tunits, &results[itest]);

          if (myrank == 0) {
               write_distribution(network_tests_list[itest], TEST_CONGESTORS, 0, &results[itest], tname, tunits);
          }

          /* allow congestors to print out results */
#ifdef VERBOSE
          for (ctest = 0; ctest < ncong_tests; ctest++) {
               print_results(config, myrank, 0, 0, " ", " ", &nullrslt);
          }
#endif

     }

     /* now print the final results table */
if (!skip_baseline) {
     print_header(config, 3, 0);
     for (itest = 0; itest < NUM_NETWORK_TESTS; itest++) {

          if (network_tests_list[itest] == P2P_LATENCY || network_tests_list[itest] == P2P_NEIGHBORS ||
              network_tests_list[itest] == ALLREDUCE_LATENCY) {
               network_test_setup(network_tests_list[itest], &ntests, &nrands, &niters, tname, tunits);

               if (network_tests_list[itest] == P2P_LATENCY || network_tests_list[itest] == ALLREDUCE_LATENCY) {
                    print_comparison_results(config, myrank, 1, 1, tname, &results_base[itest], &results[itest]);
               } else {
                    print_comparison_results(config, myrank, 1, 0, tname, &results_base[itest], &results[itest]);
               }
          }
     }
}

     return 0;
}

int congestor_test_setup(CommTest_t req_test, int a2aiters, int incastiters, int bcastiters, int allreduceiters,
                         int a2atests, int incasttests, int bcasttests, int allreducetests,
                         int *ntests, int *niters, char *tname, char *tunits)
{
     int nl=64;

     switch (req_test) {
     case A2A_CONGESTOR:
          *ntests = a2atests;
          *niters = a2aiters;
          snprintf(tname, nl, "%s (%i B)", "Alltoall", (int)(sizeof(double)*A2A_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case ALLREDUCE_CONGESTOR:
          *ntests = allreducetests;
          *niters = allreduceiters;
          snprintf(tname, nl, "%s (%i B)", "Allreduce", (int)(sizeof(double)*ALLREDUCE_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case P2P_INCAST_CONGESTOR:
          *ntests = incasttests;
          *niters = incastiters;
          snprintf(tname, nl, "%s (%i B)", "Two-sided Incast", (int)(sizeof(double)*INCAST_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case RMA_INCAST_CONGESTOR:
          *ntests = incasttests;
          *niters = incastiters;
          snprintf(tname, nl, "%s (%i B)", "Put Incast", (int)(sizeof(double)*INCAST_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case P2P_BCAST_CONGESTOR:
          *ntests = bcasttests;
          *niters = bcastiters;
          snprintf(tname, nl, "%s (%i B)", "Two-sided Bcast", (int)(sizeof(double)*BCAST_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case RMA_BCAST_CONGESTOR:
          *ntests = bcasttests;
          *niters = bcastiters;
          snprintf(tname, nl, "%s (%i B)", "Get Bcast", (int)(sizeof(double)*BCAST_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     default:
          break;
     }

     return 0;
}

int run_congestor_tests(CommConfig_t *config, int nntwk_tests, int congestor_set, MPI_Comm congestor_comm,
                        MPI_Comm test_comm, MPI_Comm local_comm, MPI_Comm global_comm)
{
     int i, iteration, itest = 0, ibar_cmplt, test_cmplt, ctest, niters, ntests, real_ntests, measure_perf;
     int a2aiters, allreduce_iters, incast_iters, bcast_iters;
     int myrank, nranks, test_myrank, test_nranks, cong_myrank, cong_nranks;
     CongestorResults_t congestor_perf;
     CommResults_t results, nullrslt;
     MPI_Request req, areq;
     double *sbuf, *rbuf;
     int nl = 64;
     char tname[nl], tunits[nl];

     mpi_error(MPI_Comm_rank(congestor_comm, &cong_myrank));
     mpi_error(MPI_Comm_size(congestor_comm, &cong_nranks));
     mpi_error(MPI_Comm_rank(local_comm, &myrank));
     mpi_error(MPI_Comm_size(local_comm, &nranks));
     mpi_error(MPI_Comm_rank(test_comm, &test_myrank));
     mpi_error(MPI_Comm_size(test_comm, &test_nranks));

     /* determine the number of iterations per pass for a2a from a base setting */
     a2aiters = A2A_BASELINE_ITERS;
     if (cong_nranks > A2A_BASE_NODES) {
          a2aiters = (int)((float)a2aiters * (float)A2A_BASE_NODES / (float)cong_nranks);
          a2aiters = (a2aiters < 1) ? 1 : a2aiters;
     }

     /* we roughly assume the allreduce algorithm behaves like a ring or recursive halving /
        recursive doubling in that the amount of data any rank sends saturates at 2 the message
        length.  In reality, message size drops per op and there is more overhead with scale.
        As a crude model we'll put a weak reduction in iters to account for this. */
     allreduce_iters = ALLREDUCE_BASELINE_ITERS;
     if (cong_nranks > ALLREDUCE_BASE_NODES) {
          allreduce_iters = (int)((float)allreduce_iters / (1.0 + log((float)cong_nranks / (float)INCAST_BASE_NODES)));
          allreduce_iters = (allreduce_iters < 1) ? 1 : allreduce_iters;
     }

     /* reduce iters for incast linearly */
     incast_iters = INCAST_BASELINE_ITERS;
     if (cong_nranks > INCAST_BASE_NODES) {
          incast_iters = (int)((float)incast_iters * (float)INCAST_BASE_NODES / (float)cong_nranks);
          incast_iters = (incast_iters < 1) ? 1 : incast_iters;
     }

     /* reduce iters for bcast linearly */
     bcast_iters = BCAST_BASELINE_ITERS;
     if (cong_nranks > BCAST_BASE_NODES) {
          incast_iters = (int)((float)bcast_iters * (float)BCAST_BASE_NODES / (float)cong_nranks);
          incast_iters = (bcast_iters < 1) ? 1 : incast_iters;
     }

     /* allocate space to the various performance tracking arrays */
     congestor_test_setup(congestor_tests_list[congestor_set], a2aiters, incast_iters, bcast_iters,
                          allreduce_iters, A2A_TESTS, INCAST_TESTS, BCAST_TESTS, ALLREDUCE_TESTS, &ntests,
                          &niters, tname, tunits);
     congestor_perf.baseline_perf_hires = malloc(sizeof(double) * ntests * niters);
     congestor_perf.perf_hires          = malloc(sizeof(double) * ntests * niters);
     if (congestor_perf.baseline_perf_hires == NULL || congestor_perf.perf_hires == NULL) {
          die("Failed to allocate perf_hires in run_congestor_tests()\n");
     }

     /* allocate the a2a buffers as needed */
     if (congestor_tests_list[congestor_set] == A2A_CONGESTOR ||
         congestor_tests_list[congestor_set] == P2P_INCAST_CONGESTOR) {
          a2a_buffers(config, congestor_comm);
     } else if (congestor_tests_list[congestor_set] == ALLREDUCE_CONGESTOR) {
         allreduce_buffers(config, congestor_comm);
     } else if (congestor_tests_list[congestor_set] == RMA_INCAST_CONGESTOR) {
          init_rma_a2a(config, congestor_comm);
     }

     /* barrier to let regular network tests get their baselines */
if (!skip_baseline && nntwk_tests) {
     print_header(config, 0, 0);
     for (itest = 0; itest < nntwk_tests; itest++) {
          print_results(config, myrank, 0, 0, " ", " ", &nullrslt);
     }
}
     mpi_error(MPI_Barrier(global_comm));

     /* loop over the congestor test and gather those baselines */
     results.distribution = NULL;
#ifdef VERBOSE
/*
 * Run the congestor baseline if it's not skipped via the -s flag or run
 * them if there are no network tests specified.
 */
if (!skip_baseline || !nntwk_tests) {
     print_header(config, 1, 0);
     for (ctest = 0; ctest < NUM_CONGESTOR_TESTS; ctest++) {

          if (ctest == congestor_set) {

               congestor_test_setup(congestor_tests_list[ctest], a2aiters, incast_iters, bcast_iters,
                                    allreduce_iters, A2A_TESTS, INCAST_TESTS, BCAST_TESTS, ALLREDUCE_TESTS,
                                    &ntests, &niters, tname, tunits);

               congestor(config, ntests, niters, congestor_comm, congestor_tests_list[ctest], 1,
                         congestor_perf.baseline_perf_hires, &congestor_perf.baseline_perf, &real_ntests);

               summarize_performance(config, congestor_perf.baseline_perf_hires,
                                     &congestor_perf.baseline_perf, real_ntests*niters, 1,
                                     0, test_comm, &results);

               print_results(config, test_myrank, 1, 0, tname, tunits, &results);
               if (test_myrank == 0) {
                    write_distribution(congestor_tests_list[ctest], TEST_NULL, 1, &results, tname, tunits);
               }
               mpi_error(MPI_Barrier(local_comm));

          } else {

               /* let the ranks in the given congestor test run in isolation */
               print_results(config, test_myrank, 0, 0, " ", " ", &nullrslt);
               mpi_error(MPI_Barrier(local_comm));

          }

     }
}
#endif

     /* sync with the network tests */
     mpi_error(MPI_Barrier(global_comm));

/* Bail now if there are no network tests. Congestor baseline only. */
if (!nntwk_tests)
    return 0;

     /* loop over the combined tests */
#ifndef VERBOSE
     print_header(config, 2, network_tests_list[itest]);
#endif
     for (itest = 0; itest < nntwk_tests; itest++) {

#ifdef VERBOSE
          print_header(config, 2, network_tests_list[itest]);
#endif
          ctest = congestor_set;

          congestor_test_setup(congestor_tests_list[ctest], a2aiters, incast_iters, bcast_iters, allreduce_iters,
                               A2A_TESTS, INCAST_TESTS, BCAST_TESTS, ALLREDUCE_TESTS, &ntests,
                               &niters, tname, tunits);

          test_cmplt = 0;
          iteration  = 0;
          ibar_cmplt = 0;
          areq       = MPI_REQUEST_NULL;
          while (test_cmplt == 0) {

               /* only measure the performance on the first iteration */
               measure_perf = (iteration == 0) ? 1 : 0;

               /* if we have done a warmup iteration, let's complete the ibarrier
                  that releases the network tests to start */
               if (iteration == 1) {
                    mpi_error(MPI_Wait(&req, MPI_STATUS_IGNORE));
               }

               congestor(config, ntests, niters, congestor_comm, congestor_tests_list[ctest], measure_perf,
                         congestor_perf.perf_hires, &congestor_perf.perf, &real_ntests);

               /* we have done a non-warmup test so we can start our ibarrier */
               if (iteration == 1) {
                    mpi_error(MPI_Ibarrier(global_comm, &req));
               }

               /* if we have started our ibarrier, use it to check if the network test
                  has finished yet.  must be consistent within our congestion test
                  communicator */
               if (iteration >= 1) {

                    /* wait on any outstanding reduction */
                    if (areq != MPI_REQUEST_NULL) {
                         mpi_error(MPI_Wait(&areq, MPI_STATUS_IGNORE));

                         /* if the network test is done we can exit this loop */
                         if (ibar_cmplt) test_cmplt = 1;
                    }

                    /* the decision that the network tests are done must be consistent across our congestor comms */
                    if (! test_cmplt) {
                         if (req != MPI_REQUEST_NULL) {
                              mpi_error(MPI_Test(&req, &ibar_cmplt, MPI_STATUS_IGNORE));
                         } else {
                              ibar_cmplt = 1;
                         }
                         mpi_error(MPI_Iallreduce(MPI_IN_PLACE, &ibar_cmplt, 1, MPI_INT, MPI_MIN, test_comm, &areq));
                    }

               }

               /* we have done warmup comms so let's release the network tests */
               if (iteration == 0) {
                    mpi_error(MPI_Ibarrier(global_comm, &req));
               }
               iteration++;
          }

          /* print out the network test results */
          print_results(config, myrank, 0, 0, " ", " ", &nullrslt);

          /* we must delay performance summary until after the network tests is done otherwise
             we introduce a stall on congesting traffic from the reductions done here */
          summarize_performance(config, congestor_perf.perf_hires, &congestor_perf.perf, real_ntests*niters, 1,
                                0, test_comm, &results);
          if (test_myrank == 0) {
               write_distribution(congestor_tests_list[ctest], network_tests_list[itest], 0, &results, tname, tunits);
          }

          for (ctest = 0; ctest < NUM_CONGESTOR_TESTS; ctest++) {
#ifdef VERBOSE
               if (ctest == congestor_set) {
                    print_results(config, test_myrank, 1, 0, tname, tunits, &results);
               } else {
                    print_results(config, test_myrank, 0, 0, " ", " ", &nullrslt);
               }
#endif
               mpi_error(MPI_Barrier(local_comm));
          }
     }

     /* now print the final results table */
/* Only print the final results when there was a baseline run to compare. */
if (!skip_baseline) {
     print_header(config, 3, 0);
     for (itest = 0; itest < nntwk_tests; itest++) {

          if (network_tests_list[itest] == P2P_LATENCY || network_tests_list[itest] == ALLREDUCE_LATENCY) {
               print_comparison_results(config, myrank, 0, 0, " ", &nullrslt, &nullrslt);
          } else if (network_tests_list[itest] == P2P_NEIGHBORS) {
               print_comparison_results(config, myrank, 0, 0, " ", &nullrslt, &nullrslt);
          }

     }
}

     free(congestor_perf.baseline_perf_hires);
     free(congestor_perf.perf_hires);

     return 0;
}

int build_subcomms(int ncongestor_tests, CommConfig_t *config, CommNodes_t *nodes, int *am_congestor, int *congestor_set,
                   MPI_Comm *congestor_comm, MPI_Comm *test_comm, MPI_Comm *local_comm, int *nt_nodes, int *nc_nodes)
{
     int i, j, ncongestor_nodes, ntest_nodes, gmyrank;
     int *allnodes, *congestor_nodes, *test_nodes;
     MPI_Comm null_comm, tmp_comm;
     FILE *fp;

     //ncongestor_nodes = (int)floor((double)(CONGESTOR_NODE_FRAC * nodes->nnodes));
     ncongestor_nodes = nodes->nnodes;
     //ntest_nodes      = nodes->nnodes - ncongestor_nodes;
     ntest_nodes      = nodes->nnodes;

     *nt_nodes        = ntest_nodes;
     *nc_nodes        = ncongestor_nodes;

     if (ntest_nodes < 2 || ncongestor_nodes < 2*NUM_CONGESTOR_TESTS) {
          if (config->myrank == 0) {
               printf("ERROR: this application must be run on at least %i nodes\n",2+(2*NUM_CONGESTOR_TESTS));
          }
          MPI_Finalize();
          exit(1);
     }

     allnodes        = malloc(sizeof(int) * nodes->nnodes);
     congestor_nodes = malloc(sizeof(int) * ncongestor_nodes);
     test_nodes      = malloc(sizeof(int) * ntest_nodes);
     if (allnodes == NULL || congestor_nodes == NULL || test_nodes == NULL) {
          die("Failed to allocate node lists in build_subcomms()\n");
     }

     /* generate a shuffled list of all nodes we will draw from.  the purpose is to prevent rank
        reordering from improving the test results and to mimic non-ideal node selection from the WLM
        observed on busy systems.  think of network tests and congestors all as different apps running
        at the same time in different allocations */
     for (i = 0; i < nodes->nnodes; i++) {
          allnodes[i] = i;
     }

     time_t now = time(NULL);
     MPI_Bcast(&now, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
     int seed = (int)now;

     /* to use a fixed seed for debug purposes, uncomment the following line */
     //seed = RSEED;
     seed = RSEED;
     shuffle(allnodes, nodes->nnodes, seed, 0);

     /* pull out the different node types */
     for (i = 0; i < ncongestor_nodes; i++) {
          congestor_nodes[i] = allnodes[i];
     }
     //j = ncongestor_nodes;
     j = 0;
     for (i = 0; i < ntest_nodes; i++) {
          test_nodes[i] = allnodes[i+j];
     }

     congestion_subcomms(config, nodes, congestor_nodes, ncongestor_nodes, am_congestor, local_comm);
     if (*am_congestor) {
          node_slice_subcomms(config, nodes, congestor_nodes, ncongestor_nodes, &tmp_comm);
     } else {
          node_slice_subcomms(config, nodes, congestor_nodes, ncongestor_nodes, &null_comm);
     }
     if (*am_congestor) {
          //node_slice_subcomms(config, nodes, test_nodes, ntest_nodes, &null_comm);
          node_slice_subcomms_t(config, nodes, test_nodes, ntest_nodes, &null_comm);
     } else {
          //node_slice_subcomms(config, nodes, test_nodes, ntest_nodes, test_comm);
          node_slice_subcomms_t(config, nodes, test_nodes, ntest_nodes, test_comm);
     }

     /* congestors now further slice up their communicator into that for the different congestor tests */
     if (*am_congestor) {
        if (ncongestor_tests)
          split_subcomms(ncongestor_tests, *local_comm, tmp_comm, congestor_set, test_comm, congestor_comm);
     }

     MPI_Comm_rank(MPI_COMM_WORLD, &gmyrank);
     if (gmyrank == 0) {

          fp = fopen("allnodes_congestion.dat", "w+");
          for (i = 0; i < nodes->nnodes; i++) {
               CommNode_t *tmp = nodes->nodes_head;
               for (j = 0; j < nodes->nnodes; j++) {
                    if (tmp->node_id == allnodes[i]) {
                         fprintf(fp, "%-32s\n", tmp->host_name);
                    }
                    tmp = tmp->next;
               }
          }
          fclose(fp);

          fp = fopen("congestors_nodes_congestion.dat", "w+");
          for (i = 0; i < ncongestor_nodes; i++) {
               CommNode_t *tmp = nodes->nodes_head;
               for (j = 0; j < nodes->nnodes; j++) {
                    if (tmp->node_id == congestor_nodes[i]) {
                         fprintf(fp, "%-32s\n", tmp->host_name);
                    }
                    tmp = tmp->next;
               }
          }
          fclose(fp);

          fp = fopen("network_nodes_congestion.dat", "w+");
          for (i = 0; i < ntest_nodes; i++) {
               CommNode_t *tmp = nodes->nodes_head;
               for (j = 0; j < nodes->nnodes; j++) {
                    if (tmp->node_id == test_nodes[i]) {
                         fprintf(fp, "%-32s\n", tmp->host_name);
                    }
                    tmp = tmp->next;
               }
          }
          fclose(fp);
     }

     free(congestor_nodes);
     free(test_nodes);
     free(allnodes);

     return 0;
}

void usage(char *cmd)
{
    printf(
"\n"
"Usage: %s [ -n <test1[,test2,...]> ] [ -c <test1[,test2,...]> ] [ -s ]\n"
"       -n <test1[,test2,...]>    network tests\n"
"       -c <test1[,test2,...]>    congestor tests\n"
"       -s                        skip initial network/congestor baseline\n"
"                                 (ignored when either no network or\n"
"                                 congestor tests are specified\n"
"\n"
"Network Tests: P2P_LATENCY\n"
"               P2P_NEIGHBORS\n"
"               ALLREDUCE_LATENCY\n"
"\n"
"Congestor Tests: A2A_CONGESTOR\n"
"                 P2P_INCAST_CONGESTOR\n"
"                 RMA_INCAST_CONGESTOR\n"
"                 RMA_BCAST_CONGESTOR\n"
"\n",
cmd);
}

int main(int argc, char* argv[])
{
     CommConfig_t test_config;
     CommNodes_t nodes;
     MPI_Comm local_comm, test_comm, congestor_comm;
     int i, c, am_congestor, congestor_set, nt_nodes, nc_nodes;
     char *str;

     while ((c = getopt (argc, argv, "hn:c:s")) != -1) {
         switch (c) {
         case 'h':
             usage(argv[0]);
             exit(0);
             break;
         case 'n':
             str = strtok(optarg, ",");
             while (str != NULL) {
                 if ((strcmp(str, "P2P_LATENCY") == 0) &&
                     !(NETWORK_TESTS & TB(P2P_LATENCY))) {
                     NETWORK_TESTS |= TB(P2P_LATENCY);
                     NUM_NETWORK_TESTS += 1;
                 } else if ((strcmp(str, "P2P_NEIGHBORS") == 0) &&
                            !(NETWORK_TESTS & TB(P2P_NEIGHBORS))) {
                     NETWORK_TESTS |= TB(P2P_NEIGHBORS);
                     NUM_NETWORK_TESTS += 1;
                 } else if ((strcmp(str, "ALLREDUCE_LATENCY") == 0) &&
                            !(NETWORK_TESTS & TB(ALLREDUCE_LATENCY))) {
                     NETWORK_TESTS |= TB(ALLREDUCE_LATENCY);
                     NUM_NETWORK_TESTS += 1;
                 } else {
                     printf("ERROR: unknown network test ('%s')\n", str);
                     exit(1);
                 }
                 str = strtok(NULL, ",");
             }
             break;
         case 'c':
             str = strtok(optarg, ",");
             while (str != NULL) {
                 if ((strcmp(str, "A2A_CONGESTOR") == 0) &&
                     !(CONGESTOR_TESTS & TB(A2A_CONGESTOR))) {
                     CONGESTOR_TESTS |= TB(A2A_CONGESTOR);
                     NUM_CONGESTOR_TESTS += 1;
                 } else if ((strcmp(str, "P2P_INCAST_CONGESTOR") == 0) &&
                            !(CONGESTOR_TESTS & TB(P2P_INCAST_CONGESTOR))) {
                     CONGESTOR_TESTS |= TB(P2P_INCAST_CONGESTOR);
                     NUM_CONGESTOR_TESTS += 1;
                 } else if ((strcmp(str, "RMA_INCAST_CONGESTOR") == 0) &&
                            !(CONGESTOR_TESTS & TB(RMA_INCAST_CONGESTOR))) {
                     CONGESTOR_TESTS |= TB(RMA_INCAST_CONGESTOR);
                     NUM_CONGESTOR_TESTS += 1;
                 } else if ((strcmp(str, "RMA_BCAST_CONGESTOR") == 0) &&
                            !(CONGESTOR_TESTS & TB(RMA_BCAST_CONGESTOR))) {
                     CONGESTOR_TESTS |= TB(RMA_BCAST_CONGESTOR);
                     NUM_CONGESTOR_TESTS += 1;
                 } else {
                     printf("ERROR: unknown congestor test ('%s')\n", str);
                     exit(1);
                 }
                 str = strtok(NULL, ",");
             }
             break;
         case 's':
             skip_baseline = true;
             break;
         }
     }

     if (!NUM_NETWORK_TESTS && !NUM_CONGESTOR_TESTS) {
         printf("ERROR: must specify network and/or congestor tests\n");
         exit(1);
     }

     network_tests_list = malloc(sizeof(CommTest_t) * NUM_NETWORK_TESTS);
     congestor_tests_list = malloc(sizeof(CommTest_t) * NUM_CONGESTOR_TESTS);

     init_mpi(&test_config, &nodes, &argc, &argv, BW_MSG_COUNT, BW_MSG_COUNT, A2A_MSG_COUNT,
              INCAST_MSG_COUNT, BCAST_MSG_COUNT, ALLREDUCE_MSG_COUNT, BW_OUTSTANDING);
     build_subcomms(NUM_CONGESTOR_TESTS, &test_config, &nodes, &am_congestor, &congestor_set, &congestor_comm,
                    &test_comm, &local_comm, &nt_nodes, &nc_nodes);
     if (am_congestor) {
         if (NUM_CONGESTOR_TESTS)
            init_rma(&test_config, congestor_comm);
     } else {
          init_rma(&test_config, test_comm);
     }

     if (test_config.myrank == 0) {
          printf("NetworkLoad Tests v%3.1f\n", VERSION);
          printf("  Test with %i MPI ranks (%i nodes)\n", test_config.nranks, nodes.nnodes);
          printf("  %i nodes running Network Tests\n", nt_nodes);
          printf("  %i nodes running Congestion Tests (min %i nodes per congestor)\n\n", nc_nodes,
                 (NUM_CONGESTOR_TESTS) ? (int)floor((double)nc_nodes/NUM_CONGESTOR_TESTS) : 0);
          printf("  Legend\n   RR = random ring communication pattern\n   Lat = latency\n   BW = bandwidth\n   BW+Sync = bandwidth with barrier");
     }

     /* define the sequence of each test type.  to increase the test count of either you must update
        the setting for NUM_NETWORK_TESTS and NUM_CONGESTOR_TESTS defined at the top */
     for (i = 0; i < NUM_NETWORK_TESTS; i++) {
         if (!NETWORK_TESTS) {
             printf("ERROR: failed to setup network tests\n");
             exit(1);
         }

         if (NETWORK_TESTS & TB(P2P_LATENCY)) {
             network_tests_list[i] = P2P_LATENCY;
             NETWORK_TESTS &= ~TB(P2P_LATENCY);
         } else if (NETWORK_TESTS & TB(P2P_NEIGHBORS)) {
             network_tests_list[i] = P2P_NEIGHBORS;
             NETWORK_TESTS &= ~TB(P2P_NEIGHBORS);
         } else if (NETWORK_TESTS & TB(ALLREDUCE_LATENCY)) {
             network_tests_list[i] = ALLREDUCE_LATENCY;
             NETWORK_TESTS &= ~TB(ALLREDUCE_LATENCY);
         }
     }

     for (i = 0; i < NUM_CONGESTOR_TESTS; i++) {
         if (!CONGESTOR_TESTS) {
             printf("ERROR: failed to setup congestor tests\n");
             exit(1);
         }

         if (CONGESTOR_TESTS & TB(A2A_CONGESTOR)) {
            congestor_tests_list[i] = A2A_CONGESTOR;
            CONGESTOR_TESTS &= ~TB(A2A_CONGESTOR);
         } else if (CONGESTOR_TESTS & TB(P2P_INCAST_CONGESTOR)) {
            congestor_tests_list[i] = P2P_INCAST_CONGESTOR;
            CONGESTOR_TESTS &= ~TB(P2P_INCAST_CONGESTOR);
         } else if (CONGESTOR_TESTS & TB(RMA_INCAST_CONGESTOR)) {
            congestor_tests_list[i] = RMA_INCAST_CONGESTOR;
            CONGESTOR_TESTS &= ~TB(RMA_INCAST_CONGESTOR);
         } else if (CONGESTOR_TESTS & TB(RMA_BCAST_CONGESTOR)) {
            congestor_tests_list[i] = RMA_BCAST_CONGESTOR;
            CONGESTOR_TESTS &= ~TB(RMA_BCAST_CONGESTOR);
         }
     }

     if (am_congestor) {

         if (NUM_CONGESTOR_TESTS) {
            run_congestor_tests(&test_config, NUM_NETWORK_TESTS, congestor_set, congestor_comm, test_comm, local_comm, MPI_COMM_WORLD);
         } else {
            CommResults_t nullrslt;
            int itest, myrank;

            mpi_error(MPI_Comm_rank(local_comm, &myrank));

            /* wait for network test baselines */
            print_header(&test_config, 0, 0);
            for (itest = 0; itest < NUM_NETWORK_TESTS; itest++) {
                 print_results(&test_config, myrank, 0, 0, " ", " ", &nullrslt);
            }
            mpi_error(MPI_Barrier(MPI_COMM_WORLD));
         }

     } else {

         if (NUM_NETWORK_TESTS) {
            run_network_tests(&test_config, NUM_CONGESTOR_TESTS, test_comm, local_comm, MPI_COMM_WORLD);
         } else {
            CommResults_t nullrslt;
            int ctest, myrank;

            mpi_error(MPI_Comm_rank(local_comm, &myrank));

            /* initial network baseline barrier */
            mpi_error(MPI_Barrier(MPI_COMM_WORLD));

            /* wait for congestor test baselines */
            print_header(&test_config, 1, 0);
            for (ctest = 0; ctest < NUM_CONGESTOR_TESTS; ctest++) {
                 print_results(&test_config, myrank, 0, 0, " ", " ", &nullrslt);
            }
            mpi_error(MPI_Barrier(MPI_COMM_WORLD));
         }

     }

     finalize_mpi(&test_config, &nodes);

     return 0;
}

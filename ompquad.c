#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <sys/time.h>


#define INITIAL_STACK_SIZE 128;  // initial stacks size


// GKQ weights
const double alpha = 0.816496580927726;
const double beta = 0.447213595499958;
static const double xgkq[12] =
{
  0.0,
  -0.942882415695480,
  -0.816496580927726,
  -0.641853342345781,
  -0.447213595499958,
  -0.236383199662150,
  0.0,
  0.236383199662150,
  0.447213595499958,
  0.641853342345781,
  0.816496580927726,
  0.942882415695480
};


// stack struct
struct stack_s {
  int el_count;            // elem count
  int el_size;             // elem size
  int mem_reserve;         // mem alloc'd for stack
  void* elements;          // pointer to stack start
};


struct my_f_params { double p1; double p2; double p3;}; // possible additional function params (not used in this demo)

typedef struct stack_s* stack_t;

typedef struct _work_t_gkq {
  double a;
  double b;
  double toler;
  double I_13;
  double fa;
  double fb;
  struct my_f_params * p;     //pointer to func params
  double (*f)(double, struct my_f_params*);
} work_gkq;



// **************************************** FUNCTION DEFS *************************************************
void create_stack(stack_t* stack, int element_size);
int empty_stack(stack_t stack);
void push_stack(stack_t stack, void* element);
void pop_stack(stack_t stack, void* element);
double gkq_adapt(stack_t stack); 
double gkq(double (*f)(double, struct my_f_params*), double a, double b,
           double TOL, struct my_f_params* p, stack_t stack);


//function called by omp-integrator
static double myfun(double x, struct my_f_params* p)
{
  // additional params can be included here
  // double p1=p->p1;
  // double p2=p->p2;
  // ...

  double sum=0;
  int i;

  //just create some load  
  for(i=0;i<5000;i++)
    sum += exp(-x)*pow(sin(x),7)*pow(cos(x),3);

  return exp(-x * x)*sum;

};

// function called by gsl integartor
static double myfun_gsl(double x, void* pv)
{
  struct my_f_params *p = (struct my_f_params*) pv;
  // additional params can be included here
  // double p1=p->p1;
  // double p2=p->p2;
  // ...  
  double sum=0;
  int i;
  //just create some load  
  for(i=0;i<5000;i++)
    sum += exp(-x)*pow(sin(x),7)*pow(cos(x),3);

  return exp(-x * x)*sum;
}


// *************************************************************************************************************
int main(int argc, char** argv)
{
  int num_threads;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();    
  }
  printf("using %d threads\n",num_threads);
  struct timeval start, end;
  // **********************************************
  int i;
  double xmin, xmax;
  double answer = 0.0;
  double reltol=1e-6; // for both schemes
  xmin = -10; //integration range
  xmax = 10;
  
  // ***********************************************
  struct my_f_params ptest;
  ptest.p1 = 0; //just as an example. Not actually used here
  ptest.p2 = 0;
  // ...


  // *************************************************
  // GSL QUADRADTURE
  // *************************************************
  int method = GSL_INTEG_GAUSS61; //51;//41;//31; //21; //15; 
  double gsltime;
  gsl_function gslfun;
  gslfun.function = &myfun_gsl;
  gslfun.params = &ptest;
  gsl_integration_workspace * gswork = NULL;
  gswork =   gsl_integration_workspace_alloc (1000);
  double integ_err;
  

  gettimeofday(&start, NULL);
  gsl_integration_qag(&gslfun, xmin, xmax, 0.0, reltol, 1000, 1,
                      gswork, &answer, &integ_err);
  gettimeofday(&end, NULL);
  gsltime = ((end.tv_sec  - start.tv_sec) * 1000000u +
             end.tv_usec - start.tv_usec) / 1.e6;

  printf("answer:%.12e, passed time:%.4e\n", answer, gsltime);
  // *************************************************
  // OMP QUADRADTURE
  // *************************************************
  stack_t stack;
  create_stack(&stack, sizeof(work_gkq));

  gettimeofday(&start, NULL);
  double integ = gkq(myfun, xmin, xmax, reltol, &ptest, stack);
  gettimeofday(&end, NULL);
  free(stack->elements);
  free(stack);

  double partime = ((end.tv_sec  - start.tv_sec) * 1000000u +
                    end.tv_usec - start.tv_usec) / 1.e6;

  printf("answer:%.12e, passed time:%.4e\n", integ, partime);

  return 0;
}



/******************************************
 * create new stack
 ******************************************/
void create_stack(stack_t* stack, int element_size)   
{
  int initial_size = INITIAL_STACK_SIZE;

  // alloc mem for stack
  (*stack) = (stack_t) malloc(sizeof(struct stack_s));
  if (!(*stack)) {
    fprintf(stderr, "error: could not allocate memory for stack.. Abort.\n");
    exit(1);
  }

  // alloc mem for elems
  (*stack)->elements = (void*) malloc(element_size * initial_size);
  (*stack)->mem_reserve = initial_size;
  if (!(*stack)->elements) {
    fprintf(stderr, "error: could not allocate memory for stack.. Abort.\n");
    exit(1);
  }

  (*stack)->el_size = element_size;
  (*stack)->el_count = 0;

}

/*****************************************
 * check if the stack is empty
 *****************************************/
int empty_stack(stack_t stack)
{
  return stack->el_count <= 0;
}


/*****************************************
 * push an element onto stack
 *****************************************/
void push_stack(stack_t stack, void* element)    //target stack and elem to be pushed
{
  int i, new_reserve;
  int log2_count;

  // do we need more mem?
  if (stack->el_count >= stack->mem_reserve)
  {
    log2_count = 0;
    for (i = stack->el_count; i > 0; i = i * 0.5) //i>>1)
    {
      log2_count++;
    }
    new_reserve = 1 << log2_count;

    // realloc and nullify
    stack->elements = (void *) realloc(stack->elements,
                                       stack->el_size * new_reserve);
    if (!stack->elements) {
      fprintf(stderr, "error: can't realloc mem.. Aborting\n");
      exit(1);
    }

    stack->mem_reserve = new_reserve;
  }

  // push elem ontp stack
  memcpy((char*)stack->elements + stack->el_count * stack->el_size,
         element, stack->el_size);
  stack->el_count++;
}


/*****************************************
 * pop an element from the stack
 *****************************************/
void pop_stack(
  stack_t stack,    
  void* element)  
{
  if (stack->el_count <= 0) {
    fprintf(stderr, "error: we're trying to pop from an empty stack.\n");
    exit(2);
  }

  stack->el_count--;
  memcpy(element,
         (char*)stack->elements + stack->el_count * stack->el_size,
         stack->el_size);
}



// **********************************************************
// *  initialize GKQ, perform first guess (still serial)
// ***************************************************************
double gkq(double (*f)(double, struct my_f_params*), double a, double b,
           double TOL, struct my_f_params* p, stack_t stack)
{
  double result = 0.0;
// *********************************************
  double m = 0.5 * (a + b);
  double h = 0.5 * (b - a);
  int i;

  double y[13];
  double fa = y[0] = f(a, p);
  double fb = y[12] = f(b, p);

  for (i = 1; i < 12; i++)         //this could be parallilized as well, but the load is low anyway
    y[i] = f(m + xgkq[i] * h, p);  


  double I_4 = (h / 6.0) * (y[0] + y[12] + 5.0 * (y[4] + y[8]));                // 4-point gauss-lobatto
  double I_7 = (h / 1470.0) * (77.0 * (y[0] + y[12]) + 432.0 * (y[2] + y[10]) + // 7-point kronrod
                               625.0 * (y[4] + y[8]) + 672.0 * y[6]);

  double I_13 = h * (0.0158271919734802 * (y[0] + y[12]) + 0.0942738402188500 * (y[1] + y[11]) + 0.155071987336585 * (y[2] + y[10]) +
                     0.188821573960182 * (y[3] + y[9]) + 0.199773405226859 * (y[4] + y[8]) + 0.224926465333340 * (y[5] + y[7]) +
                     0.242611071901408 * y[6]); //13-point Kronrod


  double Err1 = fabs(I_7 - I_13);
  double Err2 = fabs(I_4 - I_13);

  double r = (Err2 != 0.0) ? Err1 / Err2 : 1.0;
  double toler = (r > 0.0 && r < 1.0) ? TOL / r : TOL;

  if (I_13 == 0)
    I_13 = b - a;
  I_13 = fabs(I_13);


  //Prepare work and push onto stack
  work_gkq work;
  work.a = a;
  work.b = b;
  work.toler = toler;
  work.I_13 = I_13;
  work.fa = fa;
  work.fb = fb;
  work.p = p;
  work.a = a;
  work.f = f;


  push_stack(stack, &work);
  result = gkq_adapt(stack);

  return result;
}




double gkq_adapt(stack_t stack)
{
  work_gkq work;
  int ready, idle, busy;
  double integral_result = 0.0;
  busy = 0;

  #pragma omp parallel default(none) \
  shared(stack, integral_result,busy) \
  private(work, idle, ready)
  {
    ready = 0;
    idle = 1;

    while (!ready)
    {
      #pragma omp critical (stack)
      {
        if (!empty_stack(stack))
        {
          // new work
          pop_stack(stack, &work);
          if (idle)
          {
            // im busy
            busy += 1;
            idle = 0;
          }
        }
        else
        {
          // no work left
          if (!idle) {
            busy -= 1;
            idle = 1;
          }

          // all done, finish
          if (busy == 0)
          {
            ready = 1;
          }
        }
      }

      if (idle)
        continue; //if ready==1 --> leave loop

      double (*f)(double, struct my_f_params*) = work.f;

      double a = work.a;
      double b = work.b;
      double toler = work.toler;
      double I_13 = work.I_13;
      double fa = work.fa;
      double fb = work.fb;

      struct my_f_params * p = work.p;

      double m = (a + b) / 2;
      double h = (b - a) / 2;
      double mll = m - alpha * h;
      double ml = m - beta * h;
      double mr = m + beta * h;
      double mrr = m + alpha * h;

      double fmll = f(mll, p);
      double fml = f(ml, p);
      double fm = f(m, p);
      double fmr = f(mr, p);
      double fmrr = f(mrr, p);
      double I_4 = h / 6.0 * (fa + fb + 5.0 * (fml + fmr)); // 4-point Gauss-Lobatto formula.
      double I_7 = h / 1470.0 * (77.0 * (fa + fb) + 432.0 * (fmll + fmrr) + 625.0 * (fml + fmr) + 672.0 * fm);


      if (fabs(I_7 - I_4) <= toler * I_13 || mll <= a || b <= mrr)
      {
        if ((mll <= a || b <= mrr)) //Error
        {
          printf("OUT OF TOLERANCE !!!, mll:%.4e, a:%.4e, b:%.4e, mrr:%.4e,I_7-I_4:%.4e, tol:%.4e,I_13:%.4e\n",
                 mll, b, b, mrr, I_7 - I_4, toler * I_13, I_13);

        }
        #pragma omp critical (integral_result)
        {
          integral_result += I_7;
        }
      }
      else  //subdivide interval and push new work on stack
      {
        #pragma omp critical (stack)
        {
          work.a = a;
          work.b = mll;
          work.fa = fa;
          work.fb = fmll;
          push_stack(stack, &work);

          work.a = mll;
          work.b = ml;
          work.fa = fmll;
          work.fb = fml;
          push_stack(stack, &work);

          work.a = ml;
          work.b = m;
          work.fa = fml;
          work.fb = fm;
          push_stack(stack, &work);

          work.a = m;
          work.b = mr;
          work.fa = fm;
          work.fb = fmr;
          push_stack(stack, &work);

          work.a = mr;
          work.b = mrr;
          work.fa = fmr;
          work.fb = fmrr;
          push_stack(stack, &work);

          work.a = mrr;
          work.b = b;
          work.fa = fmrr;
          work.fb = fb;

          push_stack(stack, &work);

        } // pragma critical stack
      }   // else ..non-acceptable error
    } // while
  } /* end omp parallel */
  return integral_result;
}
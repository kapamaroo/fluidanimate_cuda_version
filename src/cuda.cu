//Code originally written by Richard O. Lee
//Modified by Christian Bienia and Christian Fensch

#include <stdio.h>
#include <stdlib.h>
//#include <string.h>

#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <cutil.h>

void CudaSafeCall(int lineno, cudaError_t err) {
    //    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        printf("Cuda error: line %d: %s.\n", lineno, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static inline int isLittleEndian() {
    union {
        uint16_t word;
        uint8_t byte;
    } endian_test;

    endian_test.word = 0x00FF;
    return (endian_test.byte == 0xFF);
}

union __float_and_int {
    uint32_t i;
    float    f;
};

static inline float bswap_float(float x) {
    union __float_and_int __x;

    __x.f = x;
    __x.i = ((__x.i & 0xff000000) >> 24) | ((__x.i & 0x00ff0000) >>  8) |
        ((__x.i & 0x0000ff00) <<  8) | ((__x.i & 0x000000ff) << 24);

    return __x.f;
}

static inline int bswap_int32(int x) {
    return ( (((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >>  8) |
             (((x) & 0x0000ff00) <<  8) | (((x) & 0x000000ff) << 24) );
}

////////////////////////////////////////////////////////////////////////////////

// note: icc-optimized version of this class gave 15% more
// performance than our hand-optimized SSE3 implementation

/*
class Vec3 {
 public:
    float x, y, z;

__device__    Vec3() {}
__device__ Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

__device__    float   GetLengthSq() const         { return x*x + y*y + z*z; }
__device__    float   GetLength() const           { return sqrtf(GetLengthSq()); }
__device__    Vec3 &  Normalize()                 { return *this /= GetLength(); }

__device__    Vec3 &  operator += (Vec3 const &v) { x += v.x;  y += v.y; z += v.z; return *this; }
__device__    Vec3 &  operator -= (Vec3 const &v) { x -= v.x;  y -= v.y; z -= v.z; return *this; }
__device__    Vec3 &  operator *= (float s)       { x *= s;  y *= s; z *= s; return *this; }
__device__    Vec3 &  operator /= (float s)       { x /= s;  y /= s; z /= s; return *this; }

__device__    Vec3    operator + (Vec3 const &v) const    { return Vec3(x+v.x, y+v.y, z+v.z); }
__device__    Vec3    operator - () const                 { return Vec3(-x, -y, -z); }
__device__    Vec3    operator - (Vec3 const &v) const    { return Vec3(x-v.x, y-v.y, z-v.z); }
__device__    Vec3    operator * (float s) const          { return Vec3(x*s, y*s, z*s); }
__device__    Vec3    operator / (float s) const          { return Vec3(x/s, y/s, z/s); }

__device__    float   operator * (Vec3 const &v) const    { return x*v.x + y*v.y + z*v.z; }
};
*/

typedef struct Vec3 {
    float x;
    float y;
    float z;
} Vec3;

__host__ __device__ Vec3 *operator_add (Vec3 *n,Vec3 *v,Vec3 *s)    { n->x=v->x+s->x; n->y=v->y+s->y; n->z=v->z+s->z; return n;}
__host__ __device__ Vec3 *operator_sub (Vec3 *n,Vec3 *v,Vec3 *s)    { n->x=v->x-s->x; n->y=v->y-s->y; n->z=v->z-s->z; return n;}
__host__ __device__ Vec3 *operator_mult (Vec3 *n,Vec3 *v,float s)   { n->x=v->x*s; n->y=v->y*s; n->z=v->z*s; return n;}
__host__ __device__ Vec3 *operator_div (Vec3 *n,Vec3 *v,float s)    { n->x=v->x/s; n->y=v->y/s; n->z=v->z/s; return n;}
__host__ __device__ Vec3 *operator_minus (Vec3 *n,Vec3 *v)          { n->x=-v->x; n->y=-v->y; n->z=-v->z; return n;}

__host__ __device__ float operator_mult_to_float (Vec3 *v,Vec3 *s)  { return s->x*v->x + s->y*v->y + s->z*v->z; }

__device__    float   GetLengthSq(Vec3 *v)        { return operator_mult_to_float(v,v); }
__device__    float   GetLength(Vec3 *v)          { return sqrtf(GetLengthSq(v)); }
__device__    Vec3   *Normalize(Vec3 *v)          { return operator_div(v,v,GetLength(v)); }

////////////////////////////////////////////////////////////////////////////////

// there is a current limitation of 16 particles per cell
// (this structure use to be a simple linked-list of particles but, due to
// improved cache locality, we get a huge performance increase by copying
// particles instead of referencing them)
struct Cell
{
    Vec3 p[16];
    Vec3 hv[16];
    Vec3 v[16];
    Vec3 a[16];
    float density[16];
};

////////////////////////////////////////////////////////////////////////////////

const float timeStep = 0.005f;
const float doubleRestDensity = 2000.f;
const float kernelRadiusMultiplier = 1.695f;
const float h_stiffness = 1.5f;
const float viscosity = 0.4f;

//const Vec3 externalAcceleration(0.f, -9.8f, 0.f);
const Vec3 domainMin = {-0.065f, -0.08f, -0.065f};
//const Vec3 domainMax(0.065f, 0.1f, 0.065f);

//device constants
const float externalAcceleration_x = 0.f;
const float externalAcceleration_y = -9.8f;
const float externalAcceleration_z = 0.f;

const float domainMin_x = -0.065f;
const float domainMin_y = -0.08f;
const float domainMin_z = -0.065f;

const float domainMax_x = 0.065f;
const float domainMax_y = 0.1f;
const float domainMax_z = 0.065f;


float restParticlesPerMeter;

__device__ float h;
__device__ float hSq;
__device__ float tc_orig;

__device__ float densityCoeff;
__device__ float pressureCoeff;
__device__ float viscosityCoeff;

// number of grid cells in each dimension
int h_nx;
int h_ny;
int h_nz;

__device__ int nx;
__device__ int ny;
__device__ int nz;

__device__ float delta_x;
__device__ float delta_y;
__device__ float delta_z;

int origNumParticles = 0;
int numParticles = 0;
int numCells = 0;

//device memory
__device__ Cell *cells = 0;
__device__ int *cnumPars = 0;

__device__ Cell *cells2 = 0;
__device__ int *cnumPars2 = 0;

//host memory
Cell *h_cells2 = 0;
int *h_cnumPars2 = 0;

//bool *border;			// flags which cells lie on grid boundaries

int XDIVS = 1;	// number of partitions in X
int ZDIVS = 1;	// number of partitions in Z

#define NUM_GRIDS  ((XDIVS) * (ZDIVS))

/*
struct Grid
{
    int sx, sy, sz;
    int ex, ey, ez;
} *grids;
*/

//pthread_attr_t attr;
//pthread_t *thread;
//pthread_mutex_t **mutex;	// used to lock cells in RebuildGrid and also particles in other functions
//pthread_barrier_t barrier;	// global barrier used by all threads

/*
typedef struct __thread_args {
    int tid;			//thread id, determines work partition
    int frames;			//number of frames to compute
} thread_args;			//arguments for threads
*/

////////////////////////////////////////////////////////////////////////////////

/*
 * hmgweight
 *
 * Computes the hamming weight of x
 *
 * x      - input value
 * lsb    - if x!=0 position of smallest bit set, else -1
 *
 * return - the hamming weight
 */
unsigned int hmgweight(unsigned int x, int *lsb) {
    unsigned int weight=0;
    unsigned int mask= 1;
    unsigned int count=0;

    *lsb=-1;
    while(x > 0) {
        //unsigned int temp;
        //temp=(x&mask);
        if ((x&mask) == 1) {
            weight++;
            if (*lsb == -1) *lsb = count;
        }
        x >>= 1;
        count++;
    }

    return weight;
}

void InitSim(char const *fileName, unsigned int threadnum) {
    //Compute partitioning based on square root of number of threads
    //NOTE: Other partition sizes are possible as long as XDIVS * ZDIVS == threadnum,
    //      but communication is minimal (and hence optimal) if XDIVS == ZDIVS


    FILE *file;
    int lsb;

    if (hmgweight(threadnum,&lsb) != 1) {
        printf("Number of threads must be a power of 2\n");
        exit(1);
    }

    XDIVS = 1<<(lsb/2);
    ZDIVS = 1<<(lsb/2);

    /*
    if (XDIVS*ZDIVS != threadnum) XDIVS*=2;
    assert(XDIVS * ZDIVS == threadnum);
    */

    //    thread = new pthread_t[NUM_GRIDS];
    //    grids = new struct Grid[NUM_GRIDS];

    //Load input particles

    printf("Loading file \"%s\"...\n",fileName);
    file = fopen(fileName,"rb");
    assert(file);

    fread(&restParticlesPerMeter,4,1,file);
    fread(&origNumParticles,4,1,file);

    if (!isLittleEndian()) {
        restParticlesPerMeter = bswap_float(restParticlesPerMeter);
        origNumParticles      = bswap_int32(origNumParticles);
    }
    numParticles = origNumParticles;

    printf("restParticlesPerMeter: %f\norigNumParticles: %d\n",restParticlesPerMeter,origNumParticles);

    float h_h = kernelRadiusMultiplier / restParticlesPerMeter;
    float h_hSq = h_h*h_h;
    float h_tc_orig = h_hSq*h_hSq*h_hSq;

    printf("h_h: %f\n",h_h);

    const float pi = 3.14159265358979f;

    float coeff1 = 315.f / (64.f*pi*pow(h_h,9.f));
    float coeff2 = 15.f / (pi*pow(h_h,6.f));
    float coeff3 = 45.f / (pi*pow(h_h,6.f));
    float particleMass = 0.5f*doubleRestDensity / (restParticlesPerMeter*restParticlesPerMeter*restParticlesPerMeter);

    float h_densityCoeff = particleMass * coeff1;
    float h_pressureCoeff = 3.f*coeff2 * 0.5f*h_stiffness * particleMass;
    float h_viscosityCoeff = viscosity * coeff3 * particleMass;

    //Vec3 range = domainMax - domainMin;
    float range_x = domainMax_x - domainMin_x;
    float range_y = domainMax_y - domainMin_y;
    float range_z = domainMax_z - domainMin_z;

    h_nx = (int)(range_x / h_h);
    h_ny = (int)(range_y / h_h);
    h_nz = (int)(range_z / h_h);

    assert(h_nx >= 1 && h_ny >= 1 && h_nz >= 1);

    numCells = h_nx*h_ny*h_nz;
    printf("Number of cells: %d\n",numCells);

    //Vec3 h_delta;
    float h_delta_x = range_x / h_nx;
    float h_delta_y = range_y / h_ny;
    float h_delta_z = range_z / h_nz;

    assert(h_delta_x >= h_h && h_delta_y >= h_h && h_delta_z >= h_h);
    assert(h_nx >= XDIVS && h_nz >= ZDIVS);

    /* this determines the size of the grid (in gpu world these are the blocks)
     * but as we see every block has the same size:
     * dim_x = nx / XDIVS
     * dim_y = ny          //no tiling in this dimension
     * dim_z = nz / ZDIVS

    int gi = 0;
    int sx, sz, ex, ez;
    ex = 0;
    for (int i = 0; i < XDIVS; ++i)
	{
            sx = ex;
            ex = int(float(h_nx)/float(XDIVS) * (i+1) + 0.5f);
            assert(sx < ex);

            ez = 0;
            for (int j = 0; j < ZDIVS; ++j, ++gi)
		{
                    sz = ez;
                    ez = int(float(h_nz)/float(ZDIVS) * (j+1) + 0.5f);
                    assert(sz < ez);

                    grids[gi].sx = sx;
                    grids[gi].ex = ex;
                    grids[gi].sy = 0;
                    grids[gi].ey = h_ny;
                    grids[gi].sz = sz;
                    grids[gi].ez = ez;
		}
	}
    assert(gi == NUM_GRIDS);
    */


    /* we do not need to keep information about the borders anymore,
     * every block in the GPU knows its limits from the builtin
     * variables (blockIdx, threadIdx, etc.)


    border = new bool[numCells];
    for (int i = 0; i < NUM_GRIDS; ++i)
        for (int iz = grids[i].sz; iz < grids[i].ez; ++iz)
            for (int iy = grids[i].sy; iy < grids[i].ey; ++iy)
                for (int ix = grids[i].sx; ix < grids[i].ex; ++ix)
                    {
                        int index = (iz*ny + iy)*nx + ix;
                        border[index] = false;
                        for (int dk = -1; dk <= 1; ++dk)
                            for (int dj = -1; dj <= 1; ++dj)
                                for (int di = -1; di <= 1; ++di)
                                    {
                                        int ci = ix + di;
                                        int cj = iy + dj;
                                        int ck = iz + dk;

                                        if (ci < 0) ci = 0; else if (ci > (nx-1)) ci = nx-1;
                                        if (cj < 0) cj = 0; else if (cj > (ny-1)) cj = ny-1;
                                        if (ck < 0) ck = 0; else if (ck > (nz-1)) ck = nz-1;

                                        if ( ci < grids[i].sx || ci >= grids[i].ex ||
                                            cj < grids[i].sy || cj >= grids[i].ey ||
                                            ck < grids[i].sz || ck >= grids[i].ez )
                                            border[index] = true;
                                    }
                    }
    */

    //cells = new Cell[numCells];
    //cnumPars = new int[numCells];

    CudaSafeCall( __LINE__, cudaMalloc((void**)&cells, numCells * sizeof(struct Cell)) );
    CudaSafeCall( __LINE__, cudaMalloc((void**)&cnumPars, numCells * sizeof(int)) );

    CudaSafeCall( __LINE__, cudaMalloc((void**)&cells2, numCells * sizeof(struct Cell)) );
    CudaSafeCall( __LINE__, cudaMalloc((void**)&cnumPars2, numCells * sizeof(int)) );

    //these should be tranfered to device after initialisation
    h_cells2 = (struct Cell*)malloc(numCells * sizeof(struct Cell));  //new Cell[numCells];
    h_cnumPars2 = (int*)calloc(numCells,sizeof(int));  //new int[numCells];

    assert(cells && cells2 && cnumPars && cnumPars2 && h_cells2 && h_cnumPars2);

    //we used calloc instead
    //memset(cnumPars2, 0, numCells*sizeof(int));

    float px, py, pz, hvx, hvy, hvz, vx, vy, vz;
    for (int i = 0; i < origNumParticles; ++i)
	{
            fread(&px, 4,1,file);
            fread(&py, 4,1,file);
            fread(&pz, 4,1,file);
            fread(&hvx, 4,1,file);
            fread(&hvy, 4,1,file);
            fread(&hvz, 4,1,file);
            fread(&vx, 4,1,file);
            fread(&vy, 4,1,file);
            fread(&vz, 4,1,file);
            if (!isLittleEndian()) {
                px  = bswap_float(px);
                py  = bswap_float(py);
                pz  = bswap_float(pz);
                hvx = bswap_float(hvx);
                hvy = bswap_float(hvy);
                hvz = bswap_float(hvz);
                vx  = bswap_float(vx);
                vy  = bswap_float(vy);
                vz  = bswap_float(vz);
            }

            int ci = (int)((px - domainMin.x) / h_delta_x);
            int cj = (int)((py - domainMin.y) / h_delta_y);
            int ck = (int)((pz - domainMin.z) / h_delta_z);

            if (ci < 0) ci = 0; else if (ci > (h_nx-1)) ci = nx-1;
            if (cj < 0) cj = 0; else if (cj > (h_ny-1)) cj = ny-1;
            if (ck < 0) ck = 0; else if (ck > (h_nz-1)) ck = nz-1;

            int index = (ck*h_ny + cj)*h_nx + ci;
            Cell &cell = h_cells2[index];

            int np = h_cnumPars2[index];
            if (np < 16)
		{
                    cell.p[np].x = px;
                    cell.p[np].y = py;
                    cell.p[np].z = pz;
                    cell.hv[np].x = hvx;
                    cell.hv[np].y = hvy;
                    cell.hv[np].z = hvz;
                    cell.v[np].x = vx;
                    cell.v[np].y = vy;
                    cell.v[np].z = vz;
                    ++h_cnumPars2[index];
		}
            else
                --numParticles;
	}

    fclose(file);

    CudaSafeCall ( __LINE__, cudaMemcpyToSymbol("h", &h_h, sizeof(float), 0, cudaMemcpyHostToDevice) );
    CudaSafeCall ( __LINE__, cudaMemcpyToSymbol("hSq", &h_hSq, sizeof(float), 0, cudaMemcpyHostToDevice) );
    CudaSafeCall ( __LINE__, cudaMemcpyToSymbol("densityCoeff", &h_densityCoeff, sizeof(float), 0, cudaMemcpyHostToDevice) );
    CudaSafeCall ( __LINE__, cudaMemcpyToSymbol("pressureCoeff", &h_pressureCoeff, sizeof(float), 0, cudaMemcpyHostToDevice) );
    CudaSafeCall ( __LINE__, cudaMemcpyToSymbol("viscosityCoeff", &h_viscosityCoeff, sizeof(float), 0, cudaMemcpyHostToDevice) );
    CudaSafeCall ( __LINE__, cudaMemcpyToSymbol("nx", &h_nx, sizeof(int), 0, cudaMemcpyHostToDevice) );
    CudaSafeCall ( __LINE__, cudaMemcpyToSymbol("ny", &h_ny, sizeof(int), 0, cudaMemcpyHostToDevice) );
    CudaSafeCall ( __LINE__, cudaMemcpyToSymbol("nz", &h_nz, sizeof(int), 0, cudaMemcpyHostToDevice) );
    CudaSafeCall ( __LINE__, cudaMemcpyToSymbol("delta_x", &h_delta_x, sizeof(float), 0, cudaMemcpyHostToDevice) );
    CudaSafeCall ( __LINE__, cudaMemcpyToSymbol("delta_y", &h_delta_y, sizeof(float), 0, cudaMemcpyHostToDevice) );
    CudaSafeCall ( __LINE__, cudaMemcpyToSymbol("delta_z", &h_delta_z, sizeof(float), 0, cudaMemcpyHostToDevice) );
    CudaSafeCall ( __LINE__, cudaMemcpyToSymbol("tc_orig", &h_tc_orig, sizeof(float), 0, cudaMemcpyHostToDevice) );

    printf("Number of particles: %d (%d) skipped\n",numParticles,origNumParticles-numParticles);
}

////////////////////////////////////////////////////////////////////////////////

void SaveFile(char const *fileName)
{
    printf("Saving file \"%s\"...\n", fileName);

    FILE *file;
    file = fopen(fileName,"wb+");
    assert(file);

    if (!isLittleEndian()) {
        float restParticlesPerMeter_le;
        int   origNumParticles_le;

        restParticlesPerMeter_le = bswap_float(restParticlesPerMeter);
        origNumParticles_le      = bswap_int32(origNumParticles);
        fwrite(&restParticlesPerMeter_le, 4,1,file);
        fwrite(&origNumParticles_le,      4,1,file);
    } else {
        fwrite((char *)&restParticlesPerMeter, 4,1,file);
        fwrite((char *)&origNumParticles,      4,1,file);
    }

    int count = 0;
    for (int i = 0; i < numCells; ++i)
	{
#warning reminder: we use the same host buffer for input and output
            Cell const &cell = h_cells2[i];
            int np = h_cnumPars2[i];
            for (int j = 0; j < np; ++j)
		{
                    if (!isLittleEndian()) {
                        float px, py, pz, hvx, hvy, hvz, vx,vy, vz;

                        px  = bswap_float(cell.p[j].x);
                        py  = bswap_float(cell.p[j].y);
                        pz  = bswap_float(cell.p[j].z);
                        hvx = bswap_float(cell.hv[j].x);
                        hvy = bswap_float(cell.hv[j].y);
                        hvz = bswap_float(cell.hv[j].z);
                        vx  = bswap_float(cell.v[j].x);
                        vy  = bswap_float(cell.v[j].y);
                        vz  = bswap_float(cell.v[j].z);

                        fwrite((char *)&px,  4,1,file);
                        fwrite((char *)&py,  4,1,file);
                        fwrite((char *)&pz,  4,1,file);
                        fwrite((char *)&hvx, 4,1,file);
                        fwrite((char *)&hvy, 4,1,file);
                        fwrite((char *)&hvz, 4,1,file);
                        fwrite((char *)&vx,  4,1,file);
                        fwrite((char *)&vy,  4,1,file);
                        fwrite((char *)&vz,  4,1,file);
                    } else {
                        fwrite((char *)&cell.p[j].x,  4,1,file);
                        fwrite((char *)&cell.p[j].y,  4,1,file);
                        fwrite((char *)&cell.p[j].z,  4,1,file);
                        fwrite((char *)&cell.hv[j].x, 4,1,file);
                        fwrite((char *)&cell.hv[j].y, 4,1,file);
                        fwrite((char *)&cell.hv[j].z, 4,1,file);
                        fwrite((char *)&cell.v[j].x,  4,1,file);
                        fwrite((char *)&cell.v[j].y,  4,1,file);
                        fwrite((char *)&cell.v[j].z,  4,1,file);
                    }
                    ++count;
		}
	}
    assert(count == numParticles);

    int numSkipped = origNumParticles - numParticles;
    float zero = 0.f;
    if (!isLittleEndian()) {
        zero = bswap_float(zero);
    }

    for (int i = 0; i < numSkipped; ++i)
	{
            fwrite((char *)&zero, 4,1,file);
            fwrite((char *)&zero, 4,1,file);
            fwrite((char *)&zero, 4,1,file);
            fwrite((char *)&zero, 4,1,file);
            fwrite((char *)&zero, 4,1,file);
            fwrite((char *)&zero, 4,1,file);
            fwrite((char *)&zero, 4,1,file);
            fwrite((char *)&zero, 4,1,file);
            fwrite((char *)&zero, 4,1,file);
	}

    fflush(file);
    fclose(file);
}

////////////////////////////////////////////////////////////////////////////////

void CleanUpSim()
{
    //free host memory
    free(h_cells2);
    free(h_cnumPars2);

    //free device memory
    CudaSafeCall( __LINE__, cudaFree(cells) );
    CudaSafeCall( __LINE__, cudaFree(cnumPars) );

    CudaSafeCall( __LINE__, cudaFree(cells2) );
    CudaSafeCall( __LINE__, cudaFree(cnumPars2) );
}

////////////////////////////////////////////////////////////////////////////////
#define BLOCK_BORDER_X(ix) ((ix != 0) && (ix % blockDim.x == 0))
#define BLOCK_BORDER_Y(iy) ((iy != 0) && (iy % blockDim.y == 0))
#define BLOCK_BORDER_Z(iz) ((iz != 0) && (iz % blockDim.z == 0))

#define IS_BORDER(ix,iy,iz) (BLOCK_BORDER_X(ix) || BLOCK_BORDER_Y(iy) || BLOCK_BORDER_Z(iz))

__device__ int InitNeighCellList(int ci, int cj, int ck, int *neighCells) {
    int numNeighCells = 0;

    for (int di = -1; di <= 1; ++di)
        for (int dj = -1; dj <= 1; ++dj)
            for (int dk = -1; dk <= 1; ++dk)
                {
                    int ii = ci + di;
                    int jj = cj + dj;
                    int kk = ck + dk;
                    if (ii >= 0 && ii < nx && jj >= 0 && jj < ny && kk >= 0 && kk < nz)
                        {
                            int index = (kk*ny + jj)*nx + ii;

                            //consider only cell neighbors who acltually have particles

                            if (cnumPars[index] != 0)
                                {
                                    if (IS_BORDER(ci,cj,ck)) {
                                        //pass negative value to determine the borders
                                        neighCells[numNeighCells] = -index;
                                    } else {
                                        neighCells[numNeighCells] = index;
                                    }
                                    ++numNeighCells;
                                }
                        }
                }

    return numNeighCells;
}

////////////////////////////////////////////////////////////////////////////////

__global__ void big_kernel() {

    int ix;
    int iy;
    int iz;

    ix = blockIdx.x * blockDim.x + threadIdx.x;
    iy = blockIdx.y * blockDim.y + threadIdx.y;
    iz = blockIdx.z * blockDim.z + threadIdx.z;

    printf("x: %d : %d\n",nx,blockDim.x * gridDim.x);
    printf("y: %d : %d\n",ny,blockDim.y * gridDim.y);
    printf("z: %d : %d\n",nz,blockDim.z * gridDim.z);

    //move common declarations on top

    int index = (iz*ny + iy)*nx + ix;
    int np;  //internal loop limit

    //this should be moved to shared memory
    Cell &cell = cells[index];  //just a reference to the correspondig cell //FIXME

    int neighCells[27];

    //it is safe to move the call here, neighbours do not change between the two original calls
    int numNeighCells = InitNeighCellList(ix, iy, iz, neighCells);

    //move this computation to cpu
    //const float tc_orig = hSq*hSq*hSq;

    const float parSize = 0.0002f;
    const float epsilon = 1e-10f;
    const float stiffness = 30000.f;
    const float damping = 128.f;



    ////////////////////////////////////////////////////////////////////////////////
    //void ClearParticlesMT(int i) {
    ////////////////////////////////////////////////////////////////////////////////




    //    for (int iz = grids[i].sz; iz < grids[i].ez; ++iz)
    //    for (int iy = grids[i].sy; iy < grids[i].ey; ++iy)
    //        for (int ix = grids[i].sx; ix < grids[i].ex; ++ix) {

    //    int index = (iz*ny + iy)*nx + ix;

    cnumPars[index] = 0;

    //                }  //close nested loop;



    //pthread_barrier_wait(&barrier);
    __syncthreads();



    //} close ClearParticlesMT()
    ////////////////////////////////////////////////////////////////////////////////
    //void RebuildGridMT(int i) {




    //    for (int iz = grids[i].sz; iz < grids[i].ez; ++iz)
    //        for (int iy = grids[i].sy; iy < grids[i].ey; ++iy)
    //            for (int ix = grids[i].sx; ix < grids[i].ex; ++ix) {

    //    int index = (iz*ny + iy)*nx + ix;

    Cell const &cell2 = cells2[index];
    int np2 = cnumPars2[index];
    for (int j = 0; j < np2; ++j) {
        int ci = (int)((cell2.p[j].x - domainMin_x) / delta_x);
        int cj = (int)((cell2.p[j].y - domainMin_y) / delta_y);
        int ck = (int)((cell2.p[j].z - domainMin_z) / delta_z);

        if (ci < 0) ci = 0; else if (ci > (nx-1)) ci = nx-1;
        if (cj < 0) cj = 0; else if (cj > (ny-1)) cj = ny-1;
        if (ck < 0) ck = 0; else if (ck > (nz-1)) ck = nz-1;

        int index2 = (ck*ny + cj)*nx + ci;
        // this assumes that particles cannot travel more than one grid cell per time step
        int np_renamed = cnumPars[index2];

        //use macro
        //if (border[index2]) {

        if (IS_BORDER(ck,cj,ci)) {
            //pthread_mutex_lock(&mutex[index2][0]);
            //np_renamed = cnumPars[index2]++;
            //pthread_mutex_unlock(&mutex[index2][0]);

            //use atomic
            atomicAdd(&cnumPars[index2],1);
        } else {
            cnumPars[index2]++;
        }

        //#warning what if we exceed 16 particles per cell here??
        //from what I see is that we calculate the same frame over and over
        //so every cell has at most 16 particles, from the initialisation


        Cell &cell_renamed = cells[index2];
        cell_renamed.p[np_renamed].x = cell2.p[j].x;
        cell_renamed.p[np_renamed].y = cell2.p[j].y;
        cell_renamed.p[np_renamed].z = cell2.p[j].z;
        cell_renamed.hv[np_renamed].x = cell2.hv[j].x;
        cell_renamed.hv[np_renamed].y = cell2.hv[j].y;
        cell_renamed.hv[np_renamed].z = cell2.hv[j].z;
        cell_renamed.v[np_renamed].x = cell2.v[j].x;
        cell_renamed.v[np_renamed].y = cell2.v[j].y;
        cell_renamed.v[np_renamed].z = cell2.v[j].z;
    }

    //                }  //close nested loops



    //pthread_barrier_wait(&barrier);
    __syncthreads();




    //} close RebuildGridMT()
    ////////////////////////////////////////////////////////////////////////////////
    //void InitDensitiesAndForcesMT(int i) {

    //from now on we don't change the cnumPars[index]
    np = cnumPars[index];  //internal loop limit


    //    for (int iz = grids[i].sz; iz < grids[i].ez; ++iz)
    //        for (int iy = grids[i].sy; iy < grids[i].ey; ++iy)
    //            for (int ix = grids[i].sx; ix < grids[i].ex; ++ix) {

    //    int index = (iz*ny + iy)*nx + ix;

    //    Cell &cell = cells[index];

    //    int np = cnumPars[index];

    for (int j = 0; j < np; ++j) {
        cell.density[j] = 0.f;
        cell.a[j].x = externalAcceleration_x;
        cell.a[j].y = externalAcceleration_y;
        cell.a[j].z = externalAcceleration_z;
    }


    //                }  //close nested loops



    //pthread_barrier_wait(&barrier);
    __syncthreads();




    //} close InitDensitiesAndForcesMT()
    ////////////////////////////////////////////////////////////////////////////////
    //void ComputeDensitiesMT(int i) {




    //    int neighCells[27];

    //    for (int iz = grids[i].sz; iz < grids[i].ez; ++iz)
    //        for (int iy = grids[i].sy; iy < grids[i].ey; ++iy)
    //            for (int ix = grids[i].sx; ix < grids[i].ex; ++ix) {

    //    int index = (iz*ny + iy)*nx + ix;

    //    int np = cnumPars[index];

    //    if (np == 0)  continue;
    //
    // if np==0 we do net enter the following loop

    //    int numNeighCells = InitNeighCellList(ix, iy, iz, neighCells);

    //    Cell &cell = cells[index];

    Vec3 tmp;

    for (int j = 0; j < np; ++j)
        for (int inc = 0; inc < numNeighCells; ++inc) {
            int indexNeigh = neighCells[inc];
            Cell &neigh = cells[indexNeigh];
            int numNeighPars = cnumPars[indexNeigh];
            for (int iparNeigh = 0; iparNeigh < numNeighPars; ++iparNeigh)
                if (&neigh.p[iparNeigh] < &cell.p[j]) {
                    //float distSq = (cell.p[j] - neigh.p[iparNeigh]).GetLengthSq();
                    float distSq;
                    operator_sub(&tmp,&cell.p[j],&neigh.p[iparNeigh]);
                    distSq = GetLengthSq(&tmp);
                    if (distSq < hSq) {
                        float t = hSq - distSq;
                        float tc = t*t*t;

                        //if (border[index]) {
                        if (IS_BORDER(ix,iy,iz)) {
                            //pthread_mutex_lock(&mutex[index][j]);
                            //cell.density[j] += tc;
                            //pthread_mutex_unlock(&mutex[index][j]);

                            //use atomic
                            atomicAdd(&cell.density[j],tc);
                        } else {
                            cell.density[j] += tc;
                        }

                        //if indexNeigh < 0 , cell is border
                        //if (border[indexNeigh]) {
                        if (indexNeigh<0) {
                            //pthread_mutex_lock(&mutex[-indexNeigh][iparNeigh]);
                            //neigh.density[iparNeigh] += tc;
                            //pthread_mutex_unlock(&mutex[-indexNeigh][iparNeigh]);

                            //use atomic
                            atomicAdd(&neigh.density[iparNeigh],tc);
                        } else {
                            neigh.density[iparNeigh] += tc;
                        }
                    }
                }
        }

    //                }  //close nested loops



    //pthread_barrier_wait(&barrier);
    __syncthreads();




    //} close ComputeDensitiesMT()
    ////////////////////////////////////////////////////////////////////////////////
    //void ComputeDensities2MT(int i) {




    //    const float tc = hSq*hSq*hSq;


    //    for (int iz = grids[i].sz; iz < grids[i].ez; ++iz)
    //        for (int iy = grids[i].sy; iy < grids[i].ey; ++iy)
    //            for (int ix = grids[i].sx; ix < grids[i].ex; ++ix) {

    //    int index = (iz*ny + iy)*nx + ix;

    //    Cell &cell = cells[index];

    //    int np = cnumPars[index];

    for (int j = 0; j < np; ++j) {
        cell.density[j] += tc_orig;
        cell.density[j] *= densityCoeff;
    }

    //                }  //close nested loops



    //pthread_barrier_wait(&barrier);
    __syncthreads();




    //} close ComputeDensities2MT()
    ////////////////////////////////////////////////////////////////////////////////
    //void ComputeForcesMT(int i) {




    //    int neighCells[27];

    //    for (int iz = grids[i].sz; iz < grids[i].ez; ++iz)
    //        for (int iy = grids[i].sy; iy < grids[i].ey; ++iy)
    //            for (int ix = grids[i].sx; ix < grids[i].ex; ++ix) {

    //    int index = (iz*ny + iy)*nx + ix;

    //    int np = cnumPars[index];

    //    if (np == 0)  continue;
    //
    // if np==0 we do net enter the following loop

    //    int numNeighCells = InitNeighCellList(ix, iy, iz, neighCells);

    //    Cell &cell = cells[index];

    for (int j = 0; j < np; ++j)
        for (int inc = 0; inc < numNeighCells; ++inc) {
            int indexNeigh = neighCells[inc];
            Cell &neigh = cells[indexNeigh];
            int numNeighPars = cnumPars[indexNeigh];
            for (int iparNeigh = 0; iparNeigh < numNeighPars; ++iparNeigh)
                if (&neigh.p[iparNeigh] < &cell.p[j]) {
                    //Vec3 disp = cell.p[j] - neigh.p[iparNeigh];
                    //float distSq = disp.GetLengthSq();
                    Vec3 disp;
                    operator_sub(&disp,&cell.p[j],&neigh.p[iparNeigh]);
                    float distSq = GetLengthSq(&disp);
                    if (distSq < hSq) {
                        //float dist = sqrtf(std::max(distSq, 1e-12f));
                        float dist = sqrtf(fmax(distSq, 1e-12f));
                        float hmr = h - dist;

                        //Vec3 acc = disp * pressureCoeff * (hmr*hmr/dist) *
                        //    (cell.density[j]+neigh.density[iparNeigh] - doubleRestDensity);

                        //acc += (neigh.v[iparNeigh] - cell.v[j]) * viscosityCoeff * hmr;
                        //acc /= cell.density[j] * neigh.density[iparNeigh];

                        Vec3 acc;
                        operator_mult(&acc,&disp, pressureCoeff * (hmr*hmr/dist) *
                                      (cell.density[j]+neigh.density[iparNeigh] - doubleRestDensity));

                        operator_sub(&tmp,&neigh.v[iparNeigh],&cell.v[j]);
                        operator_mult(&tmp,&tmp,viscosityCoeff * hmr);
                        operator_add(&acc,&acc,&tmp);
                        operator_div(&acc,&acc,cell.density[j] * neigh.density[iparNeigh]);

                        //if (border[index]) {
                        if (IS_BORDER(ix,iy,iz)) {
                            //pthread_mutex_lock(&mutex[index][j]);
                            //cell.a[j] += acc;
                            //pthread_mutex_unlock(&mutex[index][j]);

                            //use atomics
                            //this works because no one reads these values at the moment ??
                            atomicAdd(&cell.a[j].x,acc.x);
                            atomicAdd(&cell.a[j].y,acc.y);
                            atomicAdd(&cell.a[j].z,acc.z);
                        } else {
                            operator_add(&cell.a[j],&cell.a[j],&acc);
                        }

                        //if indexNeigh < 0 , cell is border
                        //if (border[indexNeigh]) {
                        if (indexNeigh<0) {
                            //pthread_mutex_lock(&mutex[-indexNeigh][iparNeigh]);
                            //neigh.a[iparNeigh] -= acc;
                            //pthread_mutex_unlock(&mutex[-indexNeigh][iparNeigh]);

                            //use atomics
                            //this works because no one reads these values at the moment ??

                            //#warning uncomment me
                            //atomicSub(&neigh.a[iparNeigh].x,acc.x);
                            //atomicSub(&neigh.a[iparNeigh].y,acc.y);
                            //atomicSub(&neigh.a[iparNeigh].z,acc.z);
                            atomicAdd(&neigh.a[iparNeigh].x,-acc.x);
                            atomicAdd(&neigh.a[iparNeigh].y,-acc.y);
                            atomicAdd(&neigh.a[iparNeigh].z,-acc.z);
                        } else {
                            operator_sub(&neigh.a[iparNeigh],&neigh.a[iparNeigh],&acc);
                        }
                    }
                }
        }

    //                }  //close nested loops



    //pthread_barrier_wait(&barrier);
    __syncthreads();




    //} close ComputeForcesMT()
    ////////////////////////////////////////////////////////////////////////////////
    //void ProcessCollisionsMT(int i) {




    //    const float parSize = 0.0002f;
    //    const float epsilon = 1e-10f;
    //    const float stiffness = 30000.f;
    //    const float damping = 128.f;

    //    for (int iz = grids[i].sz; iz < grids[i].ez; ++iz)
    //        for (int iy = grids[i].sy; iy < grids[i].ey; ++iy)
    //            for (int ix = grids[i].sx; ix < grids[i].ex; ++ix) {

    //    int index = (iz*ny + iy)*nx + ix;

    //    Cell &cell = cells[index];

    //    int np = cnumPars[index];

    for (int j = 0; j < np; ++j) {
        //Vec3 pos = cell.p[j] + cell.hv[j] * timeStep;
        Vec3 pos;
        operator_mult(&pos,&cell.hv[j],timeStep);
        operator_add(&pos,&pos,&cell.p[j]);

        float diff = parSize - (pos.x - domainMin_x);
        if (diff > epsilon)
            cell.a[j].x += stiffness*diff - damping*cell.v[j].x;

        diff = parSize - (domainMax_x - pos.x);
        if (diff > epsilon)
            cell.a[j].x -= stiffness*diff + damping*cell.v[j].x;

        diff = parSize - (pos.y - domainMin_y);
        if (diff > epsilon)
            cell.a[j].y += stiffness*diff - damping*cell.v[j].y;

        diff = parSize - (domainMax_y - pos.y);
        if (diff > epsilon)
            cell.a[j].y -= stiffness*diff + damping*cell.v[j].y;

        diff = parSize - (pos.z - domainMin_z);
        if (diff > epsilon)
            cell.a[j].z += stiffness*diff - damping*cell.v[j].z;

        diff = parSize - (domainMax_z - pos.z);
        if (diff > epsilon)
            cell.a[j].z -= stiffness*diff + damping*cell.v[j].z;
    }

    //                }  //close nested loops



    //pthread_barrier_wait(&barrier);
    __syncthreads();




    //} close ProcessCollisionsMT()
    ////////////////////////////////////////////////////////////////////////////////
    //void AdvanceParticlesMT(int i) {




    //    for (int iz = grids[i].sz; iz < grids[i].ez; ++iz)
    //        for (int iy = grids[i].sy; iy < grids[i].ey; ++iy)
    //            for (int ix = grids[i].sx; ix < grids[i].ex; ++ix) {

    //    int index = (iz*ny + iy)*nx + ix;

    //    Cell &cell = cells[index];

    //    int np = cnumPars[index];

    for (int j = 0; j < np; ++j) {
        //Vec3 v_half = cell.hv[j] + cell.a[j]*timeStep;
        Vec3 v_half;
        operator_mult(&v_half,&cell.a[j],timeStep);
        operator_add(&v_half,&v_half,&cell.hv[j]);

        //cell.hv[j] = v_half;
        cell.hv[j].x = v_half.x;
        cell.hv[j].y = v_half.y;
        cell.hv[j].z = v_half.z;

        //cell.v[j] *= 0.5f;
        operator_mult(&cell.v[j],&cell.v[j],0.5f);

        //cell.v[j] = cell.hv[j] + v_half;
        operator_add(&cell.v[j],&cell.hv[j],&v_half);

        //we can change v_half now, (we want to use only one tmp variable)
        //cell.p[j] += v_half * timeStep;
        operator_mult(&v_half,&v_half,timeStep);
        operator_add(&cell.p[j],&cell.p[j],&v_half);
    }

    //                }  //close nested loops



    //pthread_barrier_wait(&barrier);
    __syncthreads();




    //} close AdvanceParticlesMT()
    ////////////////////////////////////////////////////////////////////////////////




} //close big_kernel()

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
    int grid_x;
    int grid_y;
    int grid_z;

    int block_x;
    int block_y;
    int block_z;

    //minus 1, because indexing starts from 0 and here we declare the block size
    // block_x should be nx / XDIVS
    // block_y should be ny          //no partitioning here
    // block_z should be nz / ZDIVS

    grid_x = XDIVS;
    grid_y = 1;      //no partitioning here
    grid_z = ZDIVS;

    block_x = h_nx / XDIVS;
    block_y = h_ny;
    block_z = h_nz / ZDIVS;

    //should check for max grid size and block size from deviceQuery //FIXME

    //kernel stuff
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);

    if (argc < 4 || argc >= 6) {
        printf("Usage: %s <threadnum> <framenum> <.fluid input file> [.fluid output file]\n",argv[0]);
        exit(EXIT_FAILURE);
    }

    int threadnum = atoi(argv[1]);
    int framenum = atoi(argv[2]);

    //Check arguments

    if (threadnum < 1) {
        printf("<threadnum> must at least be 1\n");
        exit(EXIT_FAILURE);
    }

    if (framenum < 1) {
        printf("<framenum> must at least be 1\n");
        exit(EXIT_FAILURE);
    }

    InitSim(argv[3], threadnum);

    //move data to device
    CudaSafeCall( __LINE__, cudaMemcpy(cells2, h_cells2, numCells * sizeof(struct Cell), cudaMemcpyHostToDevice) );
    CudaSafeCall( __LINE__, cudaMemcpy(cnumPars2, h_cnumPars2, numCells * sizeof(int), cudaMemcpyHostToDevice) );


    //cuda wrapper
    for (int i = 0; i < framenum; ++i) {
        big_kernel<<<grid,block>>>();
        cudaDeviceSynchronize();
    }

    //move data to host
    /*** ATTENTION !!! we use the same host buffer ***/
    CudaSafeCall( __LINE__, cudaMemcpy(h_cells2, cells2, numCells * sizeof(struct Cell), cudaMemcpyDeviceToHost) );
    CudaSafeCall( __LINE__, cudaMemcpy(h_cnumPars2, cnumPars2, numCells * sizeof(int), cudaMemcpyDeviceToHost) );

    if (argc > 4) {
        SaveFile(argv[4]);
    }

    CleanUpSim();

    exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////

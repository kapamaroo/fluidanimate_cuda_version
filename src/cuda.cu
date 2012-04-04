//Code originally written by Richard O. Lee
//Modified by Christian Bienia and Christian Fensch
//CUDA Version by Maroudas Manolis and Petros Kalos

#include <cstdlib>
#include <cstring>

#include <iostream>
#include <fstream>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <cutil.h>

#define PARS_NUM 16

void CudaSafeCall(int lineno, cudaError_t err) {
    if( cudaSuccess != err) {
        printf("Cuda error: line %d: %s.\n", lineno, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void CUDA_CHECK_ERROR() {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        printf("Cuda error: %s.\n", cudaGetErrorString(err));
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
class Vec3 {
public:
    float x, y, z;

    __device__ __host__    Vec3() {}
    __device__ __host__    Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

    __device__ __host__    float   GetLengthSq() const         { return x*x + y*y + z*z; }
    __device__ __host__    float   GetLength() const           { return sqrtf(GetLengthSq()); }
    __device__ __host__    Vec3 &  Normalize()                 { return *this /= GetLength(); }

    __device__ __host__    Vec3 &  operator += (Vec3 const &v) { x += v.x;  y += v.y; z += v.z; return *this; }
    __device__ __host__    Vec3 &  operator -= (Vec3 const &v) { x -= v.x;  y -= v.y; z -= v.z; return *this; }
    __device__ __host__    Vec3 &  operator *= (float s)       { x *= s;  y *= s; z *= s; return *this; }
    __device__ __host__    Vec3 &  operator /= (float s)       { x /= s;  y /= s; z /= s; return *this; }

    __device__ __host__    Vec3    operator + (Vec3 const &v) const    { return Vec3(x+v.x, y+v.y, z+v.z); }
    __device__ __host__    Vec3    operator - () const                 { return Vec3(-x, -y, -z); }
    __device__ __host__    Vec3    operator - (Vec3 const &v) const    { return Vec3(x-v.x, y-v.y, z-v.z); }
    __device__ __host__    Vec3    operator * (float s) const          { return Vec3(x*s, y*s, z*s); }
    __device__ __host__    Vec3    operator / (float s) const          { return Vec3(x/s, y/s, z/s); }

    __device__ __host__    float   operator * (Vec3 const &v) const    { return x*v.x + y*v.y + z*v.z; }
};

////////////////////////////////////////////////////////////////////////////////

// there is a current limitation of PARS_NUM particles per cell
// (this structure use to be a simple linked-list of particles but, due to
// improved cache locality, we get a huge performance increase by copying
// particles instead of referencing them)
struct Cell
{
    Vec3 p[PARS_NUM];
    Vec3 hv[PARS_NUM];
    Vec3 v[PARS_NUM];
    Vec3 a[PARS_NUM];
    float density[PARS_NUM];
};

////////////////////////////////////////////////////////////////////////////////

const float timeStep = 0.005f;
const float doubleRestDensity = 2000.f;
const float kernelRadiusMultiplier = 1.695f;
const float h_stiffness = 1.5f;
const float viscosity = 0.4f;

const Vec3 domainMin(-0.065f, -0.08f, -0.065f);
const Vec3 domainMax(0.065f, 0.1f, 0.065f);

float restParticlesPerMeter;

__device__ float h;
__device__ float hSq;
__device__ float tc_orig;

__device__ float densityCoeff;
__device__ float pressureCoeff;
__device__ float viscosityCoeff;
__device__ Vec3 delta;

int origNumParticles = 0;
int numParticles = 0;
int numCells = 0;

//device memory
Cell *cells = 0;
int *cnumPars = 0;

Cell *cells2 = 0;
int *cnumPars2 = 0;

//host memory
Cell *h_cells = 0;
int *h_cnumPars = 0;

Cell *h_cells2 = 0;
int *h_cnumPars2 = 0;

// flags which cells lie on grid boundaries
//bool *h_border;
//bool *border;

int nx;
int ny;
int nz;

int XDIVS = 1;	// number of partitions in X
int ZDIVS = 1;	// number of partitions in Z

#define NUM_GRIDS  ((XDIVS) * (ZDIVS))

struct Grid {
    int sx, sy, sz;
    int ex, ey, ez;
} *grids;

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

    int lsb;

    if (hmgweight(threadnum,&lsb) != 1) {
        std::cerr << "Number of threads must be a power of 2" << std::endl;
        exit(1);
    }

    XDIVS = 1<<(lsb/2);
    ZDIVS = 1<<(lsb/2);

    if (XDIVS*ZDIVS != threadnum) XDIVS*=2;
    assert(XDIVS * ZDIVS == threadnum);

    //Load input particles
    std::cout << "Loading file \"" << fileName << "\"..." << std::endl;
    std::ifstream file(fileName, std::ios::binary);
    assert(file);

    file.read((char *)&restParticlesPerMeter, 4);
    file.read((char *)&origNumParticles, 4);
    if (!isLittleEndian()) {
        restParticlesPerMeter = bswap_float(restParticlesPerMeter);
        origNumParticles      = bswap_int32(origNumParticles);
    }
    numParticles = origNumParticles;

    float h_h = kernelRadiusMultiplier / restParticlesPerMeter;
    float h_hSq = h_h*h_h;
    float h_tc_orig = h_hSq*h_hSq*h_hSq;

    const float pi = 3.14159265358979f;

    float coeff1 = 315.f / (64.f*pi*pow(h_h,9.f));
    float coeff2 = 15.f / (pi*pow(h_h,6.f));
    float coeff3 = 45.f / (pi*pow(h_h,6.f));
    float particleMass = 0.5f*doubleRestDensity / (restParticlesPerMeter*restParticlesPerMeter*restParticlesPerMeter);

    float h_densityCoeff = particleMass * coeff1;
    float h_pressureCoeff = 3.f*coeff2 * 0.5f*h_stiffness * particleMass;
    float h_viscosityCoeff = viscosity * coeff3 * particleMass;

    Vec3 range = domainMax - domainMin;

    nx = (int)(range.x / h_h);
    ny = (int)(range.y / h_h);
    nz = (int)(range.z / h_h);

    assert(nx >= 1 && ny >= 1 && nz >= 1);

    numCells = nx * ny * nz;
    std::cout << "Number of cells: " << numCells << std::endl;

    Vec3 h_delta;
    h_delta.x = range.x / nx;
    h_delta.y = range.y / ny;
    h_delta.z = range.z / nz;

    assert(h_delta.x >= h_h && h_delta.y >= h_h && h_delta.z >= h_h);
    assert(nx >= XDIVS && nz >= ZDIVS);

    CudaSafeCall ( __LINE__, cudaMemcpyToSymbol("h", &h_h, sizeof(float), 0, cudaMemcpyHostToDevice) );
    CudaSafeCall ( __LINE__, cudaMemcpyToSymbol("hSq", &h_hSq, sizeof(float), 0, cudaMemcpyHostToDevice) );
    CudaSafeCall ( __LINE__, cudaMemcpyToSymbol("densityCoeff", &h_densityCoeff, sizeof(float), 0, cudaMemcpyHostToDevice) );
    CudaSafeCall ( __LINE__, cudaMemcpyToSymbol("pressureCoeff", &h_pressureCoeff, sizeof(float), 0, cudaMemcpyHostToDevice) );
    CudaSafeCall ( __LINE__, cudaMemcpyToSymbol("viscosityCoeff", &h_viscosityCoeff, sizeof(float), 0, cudaMemcpyHostToDevice) );
    CudaSafeCall ( __LINE__, cudaMemcpyToSymbol("delta", &h_delta, sizeof(Vec3), 0, cudaMemcpyHostToDevice) );
    CudaSafeCall ( __LINE__, cudaMemcpyToSymbol("tc_orig", &h_tc_orig, sizeof(float), 0, cudaMemcpyHostToDevice) );

    /*
    grids = new struct Grid[NUM_GRIDS];

    int gi = 0;
    int sx, sz, ex, ez;
    ex = 0;
    for (int i = 0; i < XDIVS; ++i) {
        sx = ex;
        ex = int(float(nx)/float(XDIVS) * (i+1) + 0.5f);
        assert(sx < ex);

        ez = 0;
        for (int j = 0; j < ZDIVS; ++j, ++gi) {
            sz = ez;
            ez = int(float(nz)/float(ZDIVS) * (j+1) + 0.5f);
            assert(sz < ez);

            grids[gi].sx = sx;
            grids[gi].ex = ex;
            grids[gi].sy = 0;
            grids[gi].ey = ny;
            grids[gi].sz = sz;
            grids[gi].ez = ez;
        }
    }

    assert(gi == NUM_GRIDS);

    h_border = new bool[numCells];
    assert(h_border);

    for (int i = 0; i < NUM_GRIDS; ++i)
        for (int iz = grids[i].sz; iz < grids[i].ez; ++iz)
            for (int iy = grids[i].sy; iy < grids[i].ey; ++iy)
                for (int ix = grids[i].sx; ix < grids[i].ex; ++ix)
                    {
                        int index = (iz*ny + iy)*nx + ix;
                        h_border[index] = false;
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
                                            h_border[index] = true;
                                    }
                    }
    */

    h_cells = new Cell[numCells];
    h_cnumPars = new int[numCells];

    h_cells2 = new Cell[numCells];
    h_cnumPars2 = new int[numCells];

    CudaSafeCall( __LINE__, cudaMalloc((void**)&cells, numCells * sizeof(struct Cell)) );
    CudaSafeCall( __LINE__, cudaMalloc((void**)&cnumPars, numCells * sizeof(int)) );

    CudaSafeCall( __LINE__, cudaMalloc((void**)&cells2, numCells * sizeof(struct Cell)) );
    CudaSafeCall( __LINE__, cudaMalloc((void**)&cnumPars2, numCells * sizeof(int)) );

    //CudaSafeCall( __LINE__, cudaMalloc((void**)&border, numCells * sizeof(bool)) );

    assert(h_cells && h_cnumPars);
    assert(h_cells2 && h_cnumPars2);
    assert(cells && cnumPars);
    assert(cells2 && cnumPars2);
    //assert(border);

    memset(h_cnumPars2, 0, numCells*sizeof(int));

    float px, py, pz, hvx, hvy, hvz, vx, vy, vz;
    for (int i = 0; i < origNumParticles; ++i) {
        file.read((char *)&px, 4);
        file.read((char *)&py, 4);
        file.read((char *)&pz, 4);
        file.read((char *)&hvx, 4);
        file.read((char *)&hvy, 4);
        file.read((char *)&hvz, 4);
        file.read((char *)&vx, 4);
        file.read((char *)&vy, 4);
        file.read((char *)&vz, 4);
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

        int ci = (int)((px - domainMin.x) / h_delta.x);
        int cj = (int)((py - domainMin.y) / h_delta.y);
        int ck = (int)((pz - domainMin.z) / h_delta.z);

        if (ci < 0) ci = 0; else if (ci > (nx-1)) ci = nx-1;
        if (cj < 0) cj = 0; else if (cj > (ny-1)) cj = ny-1;
        if (ck < 0) ck = 0; else if (ck > (nz-1)) ck = nz-1;

        int index = (ck*ny + cj)*nx + ci;
        Cell &cell = h_cells2[index];

        int np = h_cnumPars2[index];
        if (np < PARS_NUM) {
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

    std::cout << "Number of particles: " << numParticles << " (" << origNumParticles-numParticles << " skipped)" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////

void SaveFile(char const *fileName) {
    std::cout << "Saving file \"" << fileName << "\"..." << std::endl;

    std::ofstream file(fileName, std::ios::binary);
    assert(file);

    if (!isLittleEndian()) {
        float restParticlesPerMeter_le;
        int   origNumParticles_le;

        restParticlesPerMeter_le = bswap_float(restParticlesPerMeter);
        origNumParticles_le      = bswap_int32(origNumParticles);
        file.write((char *)&restParticlesPerMeter_le, 4);
        file.write((char *)&origNumParticles_le,      4);
    } else {
        file.write((char *)&restParticlesPerMeter, 4);
        file.write((char *)&origNumParticles,      4);
    }

    //memcpy(h_cells,    h_cells2,    numCells * sizeof(struct Cell));
    //memcpy(h_cnumPars, h_cnumPars2, numCells * sizeof(int));

    int count = 0;
    for (int i = 0; i < numCells; ++i) {
        Cell const &cell = h_cells[i];
        int np = h_cnumPars[i];
        for (int j = 0; j < np; ++j) {
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

                file.write((char *)&px,  4);
                file.write((char *)&py,  4);
                file.write((char *)&pz,  4);
                file.write((char *)&hvx, 4);
                file.write((char *)&hvy, 4);
                file.write((char *)&hvz, 4);
                file.write((char *)&vx,  4);
                file.write((char *)&vy,  4);
                file.write((char *)&vz,  4);
            } else {
                file.write((char *)&cell.p[j].x,  4);
                file.write((char *)&cell.p[j].y,  4);
                file.write((char *)&cell.p[j].z,  4);
                file.write((char *)&cell.hv[j].x, 4);
                file.write((char *)&cell.hv[j].y, 4);
                file.write((char *)&cell.hv[j].z, 4);
                file.write((char *)&cell.v[j].x,  4);
                file.write((char *)&cell.v[j].y,  4);
                file.write((char *)&cell.v[j].z,  4);
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

    for (int i = 0; i < numSkipped; ++i) {
        file.write((char *)&zero, 4);
        file.write((char *)&zero, 4);
        file.write((char *)&zero, 4);
        file.write((char *)&zero, 4);
        file.write((char *)&zero, 4);
        file.write((char *)&zero, 4);
        file.write((char *)&zero, 4);
        file.write((char *)&zero, 4);
        file.write((char *)&zero, 4);
    }
}

////////////////////////////////////////////////////////////////////////////////

void CleanUpSim() {
    //delete[] h_border;
    //delete[] grids;

    delete[] h_cells;
    delete[] h_cnumPars;

    delete[] h_cells2;
    delete[] h_cnumPars2;

    CudaSafeCall( __LINE__, cudaFree(cells) );
    CudaSafeCall( __LINE__, cudaFree(cnumPars) );

    CudaSafeCall( __LINE__, cudaFree(cells2) );
    CudaSafeCall( __LINE__, cudaFree(cnumPars2) );

    //CudaSafeCall( __LINE__, cudaFree(border) );
}

////////////////////////////////////////////////////////////////////////////////

__device__ int InitNeighCellList(int *neighCells, int *cnumPars) {
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    int cj = blockIdx.y * blockDim.y + threadIdx.y;
    int ck = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;
    int nz = blockDim.z * gridDim.z;

    int numNeighCells = 0;

    for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
            for (int dk = -1; dk <= 1; ++dk) {
                int ii = ci + di;
                int jj = cj + dj;
                int kk = ck + dk;
                if (ii >= 0 && ii < nx &&
                    jj >= 0 && jj < ny &&
                    kk >= 0 && kk < nz) {
                    int index = (kk*ny + jj)*nx + ii;

                    //consider only cell neighbors who actually have particles

                    if (cnumPars[index] != 0) {
                        neighCells[numNeighCells] = index;
                        ++numNeighCells;
                    }
                }
            }
        }
    }

    return numNeighCells;
}

////////////////////////////////////////////////////////////////////////////////

__global__ void ClearParticlesMT(int *cnumPars) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;
    //int nz = blockDim.z * gridDim.z;

    int index = (iz*ny + iy)*nx + ix;

    cnumPars[index] = 0;

} //close ClearParticlesMT()

////////////////////////////////////////////////////////////////////////////////

__global__ void RebuildGridMT(Cell *cells, int *cnumPars,Cell *cells2, int *cnumPars2) {
    const Vec3 domainMin(-0.065f, -0.08f, -0.065f);

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;
    int nz = blockDim.z * gridDim.z;

    int index = (iz*ny + iy)*nx + ix;

    Cell const &cell2 = cells2[index];
    int np2 = cnumPars2[index];
    for (int j = 0; j < np2; ++j) {
        int ci = (int)((cell2.p[j].x - domainMin.x) / delta.x);
        int cj = (int)((cell2.p[j].y - domainMin.y) / delta.y);
        int ck = (int)((cell2.p[j].z - domainMin.z) / delta.z);

        if (ci < 0) ci = 0; else if (ci > (nx-1)) ci = nx-1;
        if (cj < 0) cj = 0; else if (cj > (ny-1)) cj = ny-1;
        if (ck < 0) ck = 0; else if (ck > (nz-1)) ck = nz-1;

        int index2 = (ck*ny + cj)*nx + ci;
        // this assumes that particles cannot travel more than one grid cell per time step

        int np_renamed = cnumPars[index2];

        //we are all borders :)
        atomicAdd(&cnumPars[index2],1);

        //#warning what if we exceed PARS_NUM particles per cell here??
        //from what I see is that we calculate the same frame over and over
        //so every cell has at most PARS_NUM particles, from the initialisation

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
} //close RebuildGridMT()

////////////////////////////////////////////////////////////////////////////////

__global__ void InitDensitiesAndForcesMT(Cell *cells, int *cnumPars) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;
    //int nz = blockDim.z * gridDim.z;

    int index = (iz*ny + iy)*nx + ix;

    int np = cnumPars[index];

    const Vec3 externalAcceleration(0.f, -9.8f, 0.f);

    Cell &cell = cells[index];

    for (int j = 0; j < np; ++j) {
        cell.density[j] = 0.f;
        cell.a[j] = externalAcceleration;
    }
} //close InitDensitiesAndForcesMT()

////////////////////////////////////////////////////////////////////////////////

__global__ void ComputeDensitiesMT(Cell *cells, int *cnumPars) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;
    //int nz = blockDim.z * gridDim.z;

    int index = (iz*ny + iy)*nx + ix;

    int np = cnumPars[index];

    //    if (np == 0)  return;
    //
    // if np==0 we do net enter the following loop

    int neighCells[27];

    int numNeighCells = InitNeighCellList(neighCells, cnumPars);

    Cell &cell = cells[index];

    for (int j = 0; j < np; ++j) {
        for (int inc = 0; inc < numNeighCells; ++inc) {
            int indexNeigh = neighCells[inc];
            Cell &neigh = cells[indexNeigh];
            int numNeighPars = cnumPars[indexNeigh];
            for (int iparNeigh = 0; iparNeigh < numNeighPars; ++iparNeigh) {
                if (&neigh.p[iparNeigh] < &cell.p[j]) {
                    float distSq = (cell.p[j] - neigh.p[iparNeigh]).GetLengthSq();
                    if (distSq < hSq) {
                        float t = hSq - distSq;
                        float tc = t*t*t;

                        //we are all borders :)

                        //also consider the fact that I am neighbor of my neighbor
                        //so we both calculate the same tc.
                        //I can add tc to myself twice, because of that
                        //and no more need for atomics!

                        atomicAdd(&cell.density[j],tc);
                        atomicAdd(&neigh.density[iparNeigh],tc);

                        //cell.density[j] += 2*tc;  //FIXME ??
                    }
                }
            }
        }
    }
} //close ComputeDensitiesMT()

////////////////////////////////////////////////////////////////////////////////

__global__ void ComputeDensities2MT(Cell *cells, int *cnumPars) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;
    //int nz = blockDim.z * gridDim.z;

    int index = (iz*ny + iy)*nx + ix;

    int np = cnumPars[index];

    //move this computation to cpu
    //    const float tc_orig = hSq*hSq*hSq;

    Cell &cell = cells[index];

    for (int j = 0; j < np; ++j) {
        cell.density[j] += tc_orig;
        cell.density[j] *= densityCoeff;
    }
} //close ComputeDensities2MT()

////////////////////////////////////////////////////////////////////////////////

__global__ void ComputeForcesMT(Cell *cells, int *cnumPars) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;
    //int nz = blockDim.z * gridDim.z;

    int index = (iz*ny + iy)*nx + ix;

    int np = cnumPars[index];

    //    if (np == 0)  return;
    //
    // if np==0 we do net enter the following loop

    int neighCells[27];

    int numNeighCells = InitNeighCellList(neighCells, cnumPars);

    Cell &cell = cells[index];

    for (int j = 0; j < np; ++j) {
        for (int inc = 0; inc < numNeighCells; ++inc) {
            int indexNeigh = neighCells[inc];
            Cell &neigh = cells[indexNeigh];
            int numNeighPars = cnumPars[indexNeigh];
            for (int iparNeigh = 0; iparNeigh < numNeighPars; ++iparNeigh) {
                if (&neigh.p[iparNeigh] < &cell.p[j]) {
                    Vec3 disp = cell.p[j] - neigh.p[iparNeigh];
                    float distSq = disp.GetLengthSq();
                    if (distSq < hSq) {
                        //float dist = sqrtf(std::max(distSq, 1e-12f));
                        float dist = sqrtf(fmax(distSq, 1e-12f));
                        float hmr = h - dist;

                        Vec3 acc = disp * pressureCoeff * (hmr*hmr/dist) *
                            (cell.density[j]+neigh.density[iparNeigh] - doubleRestDensity);

                        acc += (neigh.v[iparNeigh] - cell.v[j]) * viscosityCoeff * hmr;
                        acc /= cell.density[j] * neigh.density[iparNeigh];

                        //we are all borders :)
                        //also consider the fact that I am neighbor of my neighbor
                        //so when I calculate acc, he calculates -acc
                        //I can add acc to myself twice, because of that

                        atomicAdd(&cell.a[j].x,acc.x);
                        atomicAdd(&cell.a[j].y,acc.y);
                        atomicAdd(&cell.a[j].z,acc.z);

                        atomicAdd(&neigh.a[iparNeigh].x,-acc.x);
                        atomicAdd(&neigh.a[iparNeigh].y,-acc.y);
                        atomicAdd(&neigh.a[iparNeigh].z,-acc.z);

                        //cell.a[j].x += 2*acc.x;  //FIXME ??
                        //cell.a[j].y += 2*acc.y;  //FIXME ??
                        //cell.a[j].z += 2*acc.z;  //FIXME ??
                    }
                }
            }
        }
    }
} //close ComputeForcesMT()

////////////////////////////////////////////////////////////////////////////////

__global__ void ProcessCollisionsMT(Cell *cells, int *cnumPars) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;
    //int nz = blockDim.z * gridDim.z;

    int index = (iz*ny + iy)*nx + ix;

    int np = cnumPars[index];

    const float parSize = 0.0002f;
    const float epsilon = 1e-10f;
    const float stiffness = 30000.f;
    const float damping = 128.f;
    const Vec3 domainMin(-0.065f, -0.08f, -0.065f);
    const Vec3 domainMax(0.065f, 0.1f, 0.065f);

    Cell &cell = cells[index];

    for (int j = 0; j < np; ++j) {
        Vec3 pos = cell.p[j] + cell.hv[j] * timeStep;

        float diff = parSize - (pos.x - domainMin.x);
        if (diff > epsilon)
            cell.a[j].x += stiffness*diff - damping*cell.v[j].x;

        diff = parSize - (domainMax.x - pos.x);
        if (diff > epsilon)
            cell.a[j].x -= stiffness*diff + damping*cell.v[j].x;

        diff = parSize - (pos.y - domainMin.y);
        if (diff > epsilon)
            cell.a[j].y += stiffness*diff - damping*cell.v[j].y;

        diff = parSize - (domainMax.y - pos.y);
        if (diff > epsilon)
            cell.a[j].y -= stiffness*diff + damping*cell.v[j].y;

        diff = parSize - (pos.z - domainMin.z);
        if (diff > epsilon)
            cell.a[j].z += stiffness*diff - damping*cell.v[j].z;

        diff = parSize - (domainMax.z - pos.z);
        if (diff > epsilon)
            cell.a[j].z -= stiffness*diff + damping*cell.v[j].z;
    }
} //close ProcessCollisionsMT()

////////////////////////////////////////////////////////////////////////////////

__global__ void AdvanceParticlesMT(Cell *cells, int *cnumPars) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;
    //int nz = blockDim.z * gridDim.z;

    int index = (iz*ny + iy)*nx + ix;

    int np = cnumPars[index];

    Cell &cell = cells[index];

    for (int j = 0; j < np; ++j) {
        Vec3 v_half = cell.hv[j] + cell.a[j]*timeStep;
        cell.p[j] += v_half * timeStep;
        cell.v[j] = cell.hv[j] + v_half;
        cell.v[j] *= 0.5f;
        cell.hv[j] = v_half;
    }
} //close AdvanceParticlesMT()

////////////////////////////////////////////////////////////////////////////////

void call_kernels() {
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

    block_x = nx / XDIVS;
    block_y = ny;
    block_z = nz / ZDIVS;

    //printf("grid (%d,% d, %d), block (%d, %d, %d)\n",grid_x,grid_y,grid_z,block_x,block_y,block_z);

    //kernel stuff
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);

    ClearParticlesMT          <<<grid,block>>>  (cnumPars);                                  CUDA_CHECK_ERROR();
    RebuildGridMT             <<<grid,block>>>  (cells,cnumPars,cells2,cnumPars2);           CUDA_CHECK_ERROR();
    InitDensitiesAndForcesMT  <<<grid,block>>>  (cells,cnumPars);                            CUDA_CHECK_ERROR();
    ComputeDensitiesMT        <<<grid,block>>>  (cells,cnumPars);                            CUDA_CHECK_ERROR();
    ComputeDensities2MT       <<<grid,block>>>  (cells,cnumPars);                            CUDA_CHECK_ERROR();
    ComputeForcesMT           <<<grid,block>>>  (cells,cnumPars);                            CUDA_CHECK_ERROR();
    ProcessCollisionsMT       <<<grid,block>>>  (cells,cnumPars);                            CUDA_CHECK_ERROR();
    AdvanceParticlesMT        <<<grid,block>>>  (cells,cnumPars);                            CUDA_CHECK_ERROR();
}

////////////////////////////////////////////////////////////////////////////////

void analyse_neighbors();

int main(int argc, char *argv[]) {
    if (argc < 4 || argc >= 6) {
        std::cout << "Usage: " << argv[0] << " <threadnum> <framenum> <.fluid input file> [.fluid output file]" << std::endl;
        exit(EXIT_FAILURE);
    }

    int threadnum = atoi(argv[1]);
    int framenum = atoi(argv[2]);

    //Check arguments

    if (threadnum < 1) {
        std::cerr << "<threadnum> must at least be 1" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (framenum < 1) {
        std::cerr << "<framenum> must at least be 1" << std::endl;
        exit(EXIT_FAILURE);
    }

    InitSim(argv[3], threadnum);

    //analyse_neighbors();  //debug

    //move data to device
    CudaSafeCall( __LINE__, cudaMemcpy(cells2, h_cells2, numCells * sizeof(struct Cell), cudaMemcpyHostToDevice) );
    CudaSafeCall( __LINE__, cudaMemcpy(cnumPars2, h_cnumPars2, numCells * sizeof(int), cudaMemcpyHostToDevice) );

    //CudaSafeCall( __LINE__, cudaMemcpy(border, h_border, numCells * sizeof(bool), cudaMemcpyHostToDevice) );

    for (int i = 0; i < framenum; ++i) {
        call_kernels();
    }

    //move data to host
    CudaSafeCall( __LINE__, cudaMemcpy(h_cells, cells, numCells * sizeof(struct Cell), cudaMemcpyDeviceToHost) );
    CudaSafeCall( __LINE__, cudaMemcpy(h_cnumPars, cnumPars, numCells * sizeof(int), cudaMemcpyDeviceToHost) );

    if (argc > 4)
        SaveFile(argv[4]);

    CleanUpSim();

    exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//
//                          DEBUG CODE
//
////////////////////////////////////////////////////////////////////////////////


/* Simulate neighbor relations */
void analyse_neighbors() {
    grids = new struct Grid[NUM_GRIDS];

    int gi = 0;
    int sx, sz, ex, ez;
    ex = 0;
    for (int i = 0; i < XDIVS; ++i) {
        sx = ex;
        ex = int(float(nx)/float(XDIVS) * (i+1) + 0.5f);
        assert(sx < ex);

        ez = 0;
        for (int j = 0; j < ZDIVS; ++j, ++gi) {
            sz = ez;
            ez = int(float(nz)/float(ZDIVS) * (j+1) + 0.5f);
            assert(sz < ez);

            grids[gi].sx = sx;
            grids[gi].ex = ex;
            grids[gi].sy = 0;
            grids[gi].ey = ny;
            grids[gi].sz = sz;
            grids[gi].ez = ez;
        }
    }

    assert(gi == NUM_GRIDS);

    int *symmetry = new int[numCells*27];
    memset(symmetry,0,numCells*27*sizeof(int));
    int *snum = new int[numCells];
    memset(snum,0,numCells*sizeof(int));

    for(int i = 0; i < NUM_GRIDS; ++i) {
        for(int iz = grids[i].sz; iz < grids[i].ez; ++iz) {
            for(int iy = grids[i].sy; iy < grids[i].ey; ++iy) {
                for(int ix = grids[i].sx; ix < grids[i].ex; ++ix) {
                    int sanity = 0;
                    int index = (iz*ny + iy)*nx + ix;
                    for(int dk = -1; dk <= 1; ++dk) {
                        int ck = iz + dk;
                        if(ck < 0 || ck > (nz-1)) continue;
                        for(int dj = -1; dj <= 1; ++dj) {
                            int cj = iy + dj;
                            if(cj < 0 || cj > (ny-1)) continue;
                            for(int di = -1; di <= 1; ++di) {
                                int ci = ix + di;
                                if(ci < 0 || ci > (nx-1)) continue;

                                if (h_cnumPars2[index] != 0) {
                                    int sindex = (ck*ny + cj)*nx + ci;
                                    symmetry[index*27+(snum[index]++)] = sindex;
                                    sanity++;
                                }
                            }
                        }
                    }
                    assert(sanity<=27);
                }
            }
        }
    }

    delete[] grids;

    printf("debug: neighbor status (good : bad)\n");
    for (int i=0; i<numCells; i++) {
        bool alone = true;
        for (int j=0; j<27; j++) {
            if (j>=snum[i]) break;
            for (int k=0; k<27; k++) {
                if (k>=snum[symmetry[27*i+j]]) break;
                if (i==symmetry[symmetry[27*i+j]*27+k]) alone = false;
            }
            if (alone) printf("debug: %d : %d\n",i,symmetry[27*i+j]);
        }
    }

    delete[] symmetry;
    delete[] snum;
}

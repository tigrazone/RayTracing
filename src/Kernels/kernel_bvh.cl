#include "src/utils/shared_structs.hpp"

#define MAX_RENDER_DIST 20000.0f
#define PI 3.14159265359f
#define TWO_PI 6.28318530718f
#define INV_PI 0.31830988618f
#define INV_PI_PI 0.6366197723675813f
#define PI_180 0.01745329251994f
#define INV_TWO_PI 0.15915494309f

typedef struct
{
    float3 origin;
    float3 dir;
    float3 invDir;
    int sign[3];
} Ray;

typedef struct
{
    bool hit;
    Ray ray;
    float t;
    float3 pos;
    float3 texcoord;
    float3 normal;
    const __global Triangle* object;
} IntersectData;

typedef struct
{
    __global Triangle* triangles;
    __global LinearBVHNode* nodes;
    __global Material* materials;
} Scene;

Ray InitRay(float3 origin, float3 dir)
{
    // dir = normalize(dir);

    Ray r;
    r.origin = origin;
    r.dir = dir;
    r.invDir = 1.0f / dir;
    r.sign[0] = r.invDir.x < 0;
    r.sign[1] = r.invDir.y < 0;
    r.sign[2] = r.invDir.z < 0;

    return r;
}

unsigned int HashUInt32(unsigned int x)
{
#if 0
    x = (x ^ 61) ^ (x >> 16);
    x = x + (x << 3);
    x = x ^ (x >> 4);
    x = x * 0x27d4eb2d;
    x = x ^ (x >> 15);
    return x;
#else
    return 1103515245 * x + 12345;
#endif
}

float GetRandomFloat(unsigned int* seed)
{
    *seed = (*seed ^ 61) ^ (*seed >> 16);
    *seed = *seed + (*seed << 3);
    *seed = *seed ^ (*seed >> 4);
    *seed = *seed * 0x27d4eb2d;
    *seed = *seed ^ (*seed >> 15);
    *seed = 1103515245 * (*seed) + 12345;

    return (float)(*seed) * 2.3283064365386963e-10f;
}

float3 reflect(float3 v, float3 n, float dott)
{
    return -v + dott * (n + n);
}


float3 SampleHemisphereCosine(float3 n, unsigned int* seed)
{	
    float phi = TWO_PI * GetRandomFloat(seed);
    float sinTheta2 = GetRandomFloat(seed);
    float sinTheta = native_sqrt(sinTheta2);
	
	float sincos_c, a, b;
	float sincos_s = sincos(phi, &sincos_c);
		
    float x = sincos_c * sinTheta;
    float y = sincos_s * sinTheta;
    float z = native_sqrt(1.f - sinTheta2);
	
	float3 t, s, wo;
	
	int sign = (n.z < 0.0f)* 2 -1;
	
	
		a = 1.0f / (1.0f - sign * n.z);
		b = sign * n.x * n.y * a;	
	
	wo.x = x * (1.0f - n.x * n.x * a) + y * b + z * n.x;
    wo.y = sign * ( y * (n.y * n.y * a - 1.0f) - x * b) +  + z * n.y;
    wo.z = sign * x * n.x - y * n.y + z * n.z;

    return wo;
}


bool RayTriangle(const Ray* r, const __global Triangle* triangle, IntersectData* isect)
{
    float3 e1 = triangle->v2.position - triangle->v1.position;
    float3 e2 = triangle->v3.position - triangle->v1.position;
    // Calculate planes normal vector
    float3 pvec = cross(r->dir, e2);
    float det = dot(e1, pvec);
	//9*

    // Ray is parallel to plane
    if (det < 1e-8f || -det > 1e-8f)
    {
        return false;
    }
    float inv_det = 1.0f / det;
    float3 tvec = r->origin - triangle->v1.position;
    float u = dot(tvec, pvec) * inv_det;
	
	//1/ 4*   13* 1/
    if (u < 0.0f || u > 1.0f)
    {
        return false;
    }

    float3 qvec = cross(tvec, e1);
    float v = dot(r->dir, qvec) * inv_det;
	
	//10*    23* 1/
    if (v < 0.0f || u + v > 1.0f)
    {
        return false;
    }

    float t = dot(e2, qvec) * inv_det;

	//4*    27* 1/
    if (t < isect->t)
    {
        isect->hit = true;
        isect->t = t;
        isect->pos = isect->ray.origin + isect->ray.dir * t;
        isect->object = triangle;
        isect->normal = // normalize
		(u * triangle->v2.normal + v * triangle->v3.normal + (1.0f - u - v) * triangle->v1.normal);
        isect->texcoord = u * triangle->v2.texcoord + v * triangle->v3.texcoord + (1.0f - u - v) * triangle->v1.texcoord;
    }

    return true;
}

/*
bool RayTriangle(const Ray* r, const __global Triangle* triangle, IntersectData* isect)
{
    float3 absRay = fabs(r->dir);
	int kz;
	if(absRay.x > absRay.y) {
		if(absRay.x > absRay.z) {
			kz = 0;
		} else {
			kz = 2;
		}
	} else {
		if(absRay.y > absRay.z) {
			kz = 1;
		} else {
			kz = 2;
		}
	}
	
	int kx = kz+1; if(kx == 3) kx=0;
	int ky = kx+1; if(ky == 3) ky=0;
	
	float rd[3]= {r->dir.x, r->dir.y, r->dir.z};
	
	float Sz = 1.0f / rd[kz];
	float Sx = rd[kx]*Sz;
	float Sy = rd[ky]*Sz;
	
	float A[3] = {triangle->v1.position.x - r->origin.x, triangle->v1.position.y - r->origin.y, triangle->v1.position.z - r->origin.z};
	float B[3] = {triangle->v2.position.x - r->origin.x, triangle->v2.position.y - r->origin.y, triangle->v2.position.z - r->origin.z};
	float C[3] = {triangle->v3.position.x - r->origin.x, triangle->v3.position.y - r->origin.y, triangle->v3.position.z - r->origin.z};
	
	//6*
	float Ax = A[kx] - Sx*A[kz];
	float Ay = A[ky] - Sy*A[kz];
	float Bx = B[kx] - Sx*B[kz];
	float By = B[ky] - Sy*B[kz];
	float Cx = C[kx] - Sx*C[kz];
	float Cy = C[ky] - Sy*C[kz];
	
	
	float U = Cx*By - Cy*Bx;
	if(U < 0.0f) return false;
	
	float V = Ax*Cy - Ay*Cx;
	if(V < 0.0f) return false;
	
	float W = Bx*Ay - By*Ax;
	if(W < 0.0f) return false;
	
	//12*
	float det = U + V + W;
	
	//float Az = Sz*A[kz];
	//float Bz = Sz*B[kz];
	//float Cz = Sz*C[kz];
	
	float T = Sz*(U*A[kz] + V*B[kz] + W*C[kz]);
	//16*
	
	if(T<0.0f || T >isect->t*det)
    {
        return false;
    }
	
	float inv_det = 1.0f / det;
	float t = T*inv_det;
	float u = U*inv_det;
	float v = V*inv_det;
	float w = W*inv_det;
	//20* 1/

        isect->hit = true;
        isect->t = t;
        isect->pos = isect->ray.origin + isect->ray.dir * t;
        isect->object = triangle;
        isect->normal = // normalize
		(u * triangle->v2.normal + v * triangle->v3.normal + w * triangle->v1.normal);
        isect->texcoord = u * triangle->v2.texcoord + v * triangle->v3.texcoord + w * triangle->v1.texcoord;
		
    return true;
}
*/


bool RayBounds(const __global Bounds3* bounds, const Ray* ray, float t)
{
    float t0 = max(0.0f, (bounds->pos[ray->sign[0]].x - ray->origin.x) * ray->invDir.x);
    float t1 = min(t, (bounds->pos[1 - ray->sign[0]].x - ray->origin.x) * ray->invDir.x);

    t0 = max(t0, (bounds->pos[ray->sign[1]].y - ray->origin.y) * ray->invDir.y);
    t1 = min(t1, (bounds->pos[1 - ray->sign[1]].y - ray->origin.y) * ray->invDir.y);

    t0 = max(t0, (bounds->pos[ray->sign[2]].z - ray->origin.z) * ray->invDir.z);
    t1 = min(t1, (bounds->pos[1 - ray->sign[2]].z - ray->origin.z) * ray->invDir.z);

    return (t1 >= t0);

}

IntersectData Intersect(Ray *ray, const Scene* scene)
{
    IntersectData isect;
    isect.hit = false;
    isect.ray = *ray;
    isect.t = MAX_RENDER_DIST;
    
    float t;
    // Follow ray through BVH nodes to find primitive intersections
    int toVisitOffset = 0, currentNodeIndex = 0;
    int nodesToVisit[64];
    while (true)
    {
        __global LinearBVHNode* node = &scene->nodes[currentNodeIndex];

        if (RayBounds(&node->bounds, ray, isect.t))
        {
            // Leaf node
            if (node->nPrimitives > 0)
            {
                // Intersect ray with primitives in leaf BVH node
                for (int i = 0; i < node->nPrimitives; ++i)
                {
                    RayTriangle(ray, &scene->triangles[node->offset + i], &isect);
                }

                if (!toVisitOffset) break;
                currentNodeIndex = nodesToVisit[--toVisitOffset];
            }
            else
            {
                // Put far BVH node on _nodesToVisit_ stack, advance to near node
                if (ray->sign[node->axis])
                {
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node->offset;
                }
                else
                {
                    nodesToVisit[toVisitOffset++] = node->offset;
                    currentNodeIndex = currentNodeIndex + 1;
                }
            }
        }
        else
        {
            if (!toVisitOffset) break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }

    return isect;
}

//#define smp ((sampler_t)(CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR))

float3 SampleSky(__read_only image2d_t tex, float3 dir)
{
    //return 0.0f;
    const sampler_t smp = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;

    // Convert (normalized) dir to spherical coordinates.
    float2 coords = (float2)(atan2(dir.x, dir.y) + PI, acos(dir.z)* INV_PI);
    coords.x = (coords.x < 0.0f ? coords.x + TWO_PI : coords.x) * INV_TWO_PI;
	/*
    coords.x *= INV_TWO_PI;
    coords.y *= INV_PI;
*/
    return read_imagef(tex, smp, coords).xyz;

}

float DistributionGGX(float cosTheta, float alpha)
{
    const float alpha2 = alpha*alpha;
	const float zzz = cosTheta * cosTheta * (alpha2 - 1.0f) + 1.0f;
    return alpha2 * INV_PI / (zzz * zzz);
}

float3 SampleGGX(float3 n, float alpha, float* cosTheta, unsigned int* seed)
{
    //float phi = TWO_PI * GetRandomFloat(seed);
    float xi = GetRandomFloat(seed);
    float cosTheta2 = (1.0f - xi) / (xi * (alpha * alpha - 1.0f) + 1.0f);
    *cosTheta = native_sqrt(cosTheta2);
    float sinTheta = native_sqrt(1.0f - cosTheta2);
		
	float sincos_c, a, b;
	float sincos_s = sincos(TWO_PI * GetRandomFloat(seed), &sincos_c);
		
    float x = sincos_c * sinTheta;
    float y = sincos_s * sinTheta;
    float z = *cosTheta;
	
	float3 t, s, wo;
	
	int sign = (n.z < 0.0f)* 2 -1;
	
	
		a = 1.0f / (1.0f - sign * n.z);
		b = sign * n.x * n.y * a;
	
	wo.x = x * (1.0f - n.x * n.x * a) + y * b + z * n.x;
    wo.y = sign * ( y * (n.y * n.y * a - 1.0f) - x * b) +  + z * n.y;
    wo.z = sign * x * n.x - y * n.y + z * n.z;

    return wo;
	
	/*	
    float phi = TWO_PI * GetRandomFloat(seed);
    float sinTheta2 = GetRandomFloat(seed);
    float sinTheta = native_sqrt(sinTheta2);
	
	float sincos_c, a, b;
	float sincos_s = sincos(phi, &sincos_c);
		
    float x = sincos_c * sinTheta;
    float y = sincos_s * sinTheta;
    float z = native_sqrt(1.f - sinTheta2);
	
	float3 t, s, wo;
	
	int sign = (n.z < 0.0f)* 2 -1;
	
	
		a = 1.0f / (1.0f - sign * n.z);
		b = sign * n.x * n.y * a;	
	
	wo.x = x * (1.0f - n.x * n.x * a) + y * b + z * n.x;
    wo.y = sign * ( y * (n.y * n.y * a - 1.0f) - x * b) +  + z * n.y;
    wo.z = sign * x * n.x - y * n.y + z * n.z;
	*/
}


//3* -> 1*
float DistributionGGX1(float cosTheta2, float alpha2)
{
	const float zzz = cosTheta2 * alpha2 + 1.0f;
    return (1.0f + alpha2) * INV_PI / (zzz * zzz);
}

float3 SampleGGX1(float3 n, float alpha2, float* cosTheta, float* cosTheta2, unsigned int* seed)
{
    //float phi = TWO_PI * GetRandomFloat(seed);
    float xi = GetRandomFloat(seed);
    *cosTheta2 = (1.0f - xi) / (xi * alpha2 + 1.0f);
    *cosTheta = native_sqrt(*cosTheta2);
    float sinTheta = native_sqrt(1.0f - *cosTheta2);
		
	float sincos_c, a, b;
	float sincos_s = sincos(TWO_PI * GetRandomFloat(seed), &sincos_c);
		
    float x = sincos_c * sinTheta;
    float y = sincos_s * sinTheta;
    float z = *cosTheta;
	
	float3 t, s, wo;
	
	int sign = (n.z < 0.0f)* 2 -1;
	
	
		a = 1.0f / (1.0f - sign * n.z);
		b = sign * n.x * n.y * a;
	
	wo.x = x * (1.0f - n.x * n.x * a) + y * b + z * n.x;
    wo.y = sign * ( y * (n.y * n.y * a - 1.0f) - x * b) +  + z * n.y;
    wo.z = sign * x * n.x - y * n.y + z * n.z;

    return wo;
}

//not used
/*
float FresnelShlick(float f0, float nDotWi)
{
    return f0 + (1.0f - f0) * pow(1.0f - nDotWi, 5.0f);
}
*/

float3 SampleDiffuse(float3 wo, float3* wi, float* pdf1, float3 texcoord, float3 normal, const __global Material* material, unsigned int* seed)
{
    *wi = SampleHemisphereCosine(normal, seed);
    // *pdf = dot(*wi, normal) * INV_PI;
	*pdf1 = PI / dot(*wi, normal);

	//checker texture
    float3 albedo = (sin(texcoord.x * 64) > 0) * (sin(texcoord.y * 64) > 0) + (sin(texcoord.x * 64 + PI) > 0) * (sin(texcoord.y * 64 + PI) > 0) * 2.0f;
    return albedo * material->diffuse * INV_PI;
}

float3 SampleDiffuse2(float3 wo, float3* wi, float* pdf1, float3 texcoord, float3 normal, const __global Material* material, unsigned int* seed)
{
    *wi = SampleHemisphereCosine(normal, seed);
    // *pdf = dot(*wi, normal) * INV_PI;
	*pdf1 = PI / dot(*wi, normal);

	//checker texture
    float3 albedo = (sin(texcoord.x * 64) > 0) * (sin(texcoord.y * 64) > 0) + (sin(texcoord.x * 64 + PI) > 0) * (sin(texcoord.y * 64 + PI) > 0) * 2.0f;
    return albedo * material->diffuse * INV_PI_PI;
}

float3 SampleSpecular(float3 wo, float3* wi, float* pdf1, float3 normal, const __global Material* material, unsigned int* seed)
{
    float cosTheta, cosTheta2;
	
    float alpha2 = material->roughness*material->roughness -1.0f;
    float3 wh = SampleGGX1(normal, alpha2, &cosTheta, &cosTheta2, seed);
	
	float dott = dot(wo, wh);
	dott += dott;
    *wi = //reflect(wo, wh, dott); 
		-wo + dott * wh;
	float dots = dot(*wi, normal) * dot(wo, normal); 
    if (dots < 0.0f) return 0.0f;
	
    float D = DistributionGGX1(cosTheta2, alpha2);
	
    //*pdf = D * cosTheta / (4.0f * dott);
    *pdf1 = (dott + dott) / (D * cosTheta);
	
    // Actually, _material->ior_ isn't ior value, this is f0 value for now
    return D / (dots + dots + dots + dots) * material->specular;
}

float3 SampleSpecular2(float3 wo, float3* wi, float* pdf1, float3 normal, const __global Material* material, unsigned int* seed)
{
    float cosTheta, cosTheta2;
	
    float alpha2 = material->roughness*material->roughness -1.0f;
    float3 wh = SampleGGX1(normal, alpha2, &cosTheta, &cosTheta2, seed);
	
	float dott = dot(wo, wh);
	dott += dott;
    *wi = //reflect(wo, wh, dott); 
		-wo + dott * wh;
	float dots = dot(*wi, normal) * dot(wo, normal); 
    if (dots < 0.0f) return 0.0f;
	
    float D = DistributionGGX1(cosTheta2, alpha2);
	
    //*pdf = D * cosTheta / (4.0f * dott);
    *pdf1 = (dott + dott) / (D * cosTheta);
	
    // Actually, _material->ior_ isn't ior value, this is f0 value for now
    return D / (dots + dots) * material->specular;
}

float3 SampleBrdf(float3 wo, float3* wi, float* pdf, float3 texcoord, float3 normal, const __global Material* material, unsigned int* seed)
{
    bool doSpecular = (material->specular.x + material->specular.y + material->specular.z) > 0.0f;
    bool doDiffuse =  (material->diffuse.x + material->diffuse.y + material->diffuse.z) > 0.0f;

    if (doSpecular && !doDiffuse)
    {
        return SampleSpecular(wo, wi, pdf, normal, material, seed);
    }
    else if (!doSpecular && doDiffuse)
    {
        return SampleDiffuse(wo, wi, pdf, texcoord, normal, material, seed);
    }
    else if (doSpecular && doDiffuse)
    {
        if (GetRandomFloat(seed) > 0.5f)
        {
            return SampleSpecular2(wo, wi, pdf, normal, material, seed);
        }
        else
        {
            return SampleDiffuse2(wo, wi, pdf, texcoord, normal, material, seed);
        }
    }
    else
    {
        return 0.0f;
    }

}

float3 RenderPT(Ray* ray, const Scene* scene, unsigned int* seed, __read_only image2d_t tex)
{
    float3 radiance = 0.0f;
    float3 beta = 1.0f;
	
	int rrDepth = 8;
	int minDepth = 5;
            
    for (int i = 0; i<rrDepth ; ++i)
    {
        IntersectData isect = Intersect(ray, scene);

        if (!isect.hit)
        {
            float3 val = beta * SampleSky(tex, ray->dir);
            radiance += val;
            break;
        }
        
        const __global Material* material = &scene->materials[isect.object->mtlIndex];
        radiance += beta * material->emission * 50.0f;

        float3 wi;
        float3 wo = -ray->dir;
        float pdf = 0.0f;
        float3 f = SampleBrdf(wo, &wi, &pdf, isect.texcoord, isect.normal, material, seed);
        if (pdf <= 0.0f) break;

		//был /
        float3 mul = f * dot(wi, isect.normal) * pdf;
        beta *= mul;
        *ray = InitRay(isect.pos + wi * 0.01f, wi);
		
		if (i >= minDepth) {
				// Randomly terminate a path with a probability inversely equal to the max reflection
				float p = max(radiance.x, max(radiance.y, radiance.z));
				if (GetRandomFloat(seed) > p)
					break;
				
				radiance /= p;
		}
    }
    
    return max(radiance, 0.0f);
}

float2 PointInHexagon(unsigned int* seed)
{
    float2 hexPoints[3] = { (float2)(-1.0f, 0.0f), (float2)(0.5f, 0.866f), (float2)(0.5f, -0.866f) };
    int x = floor(GetRandomFloat(seed) * 3.0f);
    float2 v1 = hexPoints[x];
    float2 v2 = hexPoints[(x + 1) % 3];
    float p1 = GetRandomFloat(seed);
    float p2 = GetRandomFloat(seed);
    return (float2)(p1 * v1.x + p2 * v2.x, p1 * v1.y + p2 * v2.y);
}




Ray CreateRay(float3 cameraPos, float3 cameraFront, float3 cameraUp, float3 cameraX, float4 extra_data, unsigned int* seed)
{
    float x = (float)(get_global_id(0) % (uint)(1.0f / extra_data.x)) + GetRandomFloat(seed) - 0.5f;
    float y = (float)(get_global_id(0) * extra_data.x) + GetRandomFloat(seed) - 0.5f;

    x = ( ((x + x + 1.0f) * extra_data.x) - 1) * extra_data.w;
    y = ( ((y + y + 1.0f) * extra_data.y) - 1) * extra_data.z;

    float3 dir = // normalize
	(x * cameraX + y * cameraUp + cameraFront);
	/*
    // Simple Depth of Field
    float3 pointAimed = cameraPos + 60.0f * dir;
    //float2 dofDir = (float2)(GetRandomFloat(seed), GetRandomFloat(seed));
    //dofDir = normalize(dofDir * 2.0f - 1.0f) * GetRandomFloat(seed);
    float2 dofDir = PointInHexagon(seed);
    float r = 1.0f;
    float3 newPos = cameraPos + dofDir.x * r * cross(cameraFront, cameraUp) + dofDir.y * r * cameraUp;
    
    return InitRay(newPos, normalize(pointAimed - newPos));
	*/
    return InitRay(cameraPos, dir);
}

#define GAMMA_CORRECTION
#define ONE_GAMMA 0.45454545454f

float3 ToGamma(float3 value)
{
#ifdef GAMMA_CORRECTION
    //return pow(value, 1.0f / 2.2f);
    return native_powr(value, ONE_GAMMA);
#else
    return value;
#endif
}

float3 FromGamma(float3 value)
{
#ifdef GAMMA_CORRECTION
    return native_powr(value, 2.2f);
#else
    return value;
#endif
}

__kernel void KernelEntry
(
    // Output
    // Input
    __global float3* result,
    __global Triangle* triangles,
    __global LinearBVHNode* nodes,
    __global Material* materials,
    uint width,
    uint height,
    float3 cameraPos,
    float3 cameraFront,
    float3 cameraUp,
    float3 cameraX,
    float4 extra_data,
    unsigned int frameCount,
    unsigned int frameSeed,
    __read_only image2d_t tex
)
{
    Scene scene = { triangles, nodes, materials };

    unsigned int seed = get_global_id(0) + HashUInt32(frameCount);
    
    Ray ray = CreateRay(cameraPos, cameraFront, cameraUp, cameraX, extra_data, &seed);
    float3 radiance = RenderPT(&ray, &scene, &seed, tex);
        
    if (frameCount == 0)
    {
        result[get_global_id(0)] = ToGamma(radiance);
    }
    else
    {
        result[get_global_id(0)] = ToGamma((FromGamma(result[get_global_id(0)]) * (frameCount - 1) + radiance) / frameCount);
    }

}

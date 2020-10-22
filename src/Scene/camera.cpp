#include "camera.hpp"
#include "renderers/render.hpp"
#include "io/inputsystem.hpp"
#include <Windows.h>
#include <iostream>


#define PI_180 0.01745329251994f

Camera::Camera(std::shared_ptr<Viewport> viewport)
    :
    m_Viewport(viewport),
    m_Origin(0.0f, -20.0f, 20.0f),
    m_Pitch(MATH_PIDIV2),
    m_Yaw(MATH_PIDIV2),
    m_Speed(32.0f),
    m_FrameCount(0),
    m_Up(0.0f, 0.0f, 1.0f)
{
	int frontback = -1, strafe = 0;
	
	for(int i=0;i<5;i++)
	{
		m_Origin += float3(std::cos(m_Yaw) * std::sin(m_Pitch) * frontback - std::cos(m_Yaw - MATH_PIDIV2) * strafe,
			std::sin(m_Yaw) * std::sin(m_Pitch) * frontback - std::sin(m_Yaw - MATH_PIDIV2) * strafe, std::cos(m_Pitch) * frontback) * m_Speed * render->GetDeltaTime();
		m_Front = float3(std::cos(m_Yaw) * std::sin(m_Pitch), std::sin(m_Yaw) * std::sin(m_Pitch), std::cos(m_Pitch));
	}
	
    float3 right = Cross(m_Front, m_Up).Normalize();
    float3 up = Cross(right, m_Front);
	
	time0 = clock();
}

void Camera::Update()
{
    static POINT point = { 0, 0 };
    unsigned short x, y;
    input->GetMousePos(&x, &y);
    POINT mouseClient = { x, y };
    ScreenToClient(render->GetHWND(), &mouseClient);

    if (input->IsMousePressed(MK_RBUTTON) && mouseClient.x > 0 && mouseClient.y > 0 && mouseClient.x < m_Viewport->width && mouseClient.y < m_Viewport->height)
    {
        float sensivity = 0.00075f;
        m_Yaw -= (x - point.x) * sensivity;
        m_Pitch += (y - point.y) * sensivity;
        float epsilon = 0.0001f;
        m_Pitch = clamp(m_Pitch, 0.0f + epsilon, MATH_PI - epsilon);
        input->SetMousePos(point.x, point.y);
        m_FrameCount = 0;
		time0 = clock();
    }
    else
    {
        point = { x, y };
    }

    int frontback = input->IsKeyDown('W') - input->IsKeyDown('S');
    int strafe = input->IsKeyDown('A') - input->IsKeyDown('D');

    if (frontback != 0 || strafe != 0)
    {
        m_FrameCount = 0;
		time0 = clock();
    }
    
    m_Origin += float3(std::cos(m_Yaw) * std::sin(m_Pitch) * frontback - std::cos(m_Yaw - MATH_PIDIV2) * strafe,
        std::sin(m_Yaw) * std::sin(m_Pitch) * frontback - std::sin(m_Yaw - MATH_PIDIV2) * strafe, std::cos(m_Pitch) * frontback) * m_Speed * render->GetDeltaTime();
    m_Front = float3(std::cos(m_Yaw) * std::sin(m_Pitch), std::sin(m_Yaw) * std::sin(m_Pitch), std::cos(m_Pitch));

    float3 right = Cross(m_Front, m_Up).Normalize();
    float3 up = Cross(right, m_Front);
    float3 camX = Cross(m_Front, up);
	
	int width = m_Viewport->width;
	int height = m_Viewport->height;
	
    float fov = 45.0f * PI_180;
    float angle = tan(0.5f * fov);
	float aspectratio = (float)(width) / (float)(height);
	
	float4 extra_data = float4(1.0f / (float)(width), 1.0f / (float)(height), angle, angle*aspectratio);

    render->GetCLKernel()->SetArgument(RenderKernelArgument_t::CAM_ORIGIN, &m_Origin, sizeof(float3));
    render->GetCLKernel()->SetArgument(RenderKernelArgument_t::CAM_FRONT, &m_Front, sizeof(float3));
    render->GetCLKernel()->SetArgument(RenderKernelArgument_t::CAM_UP, &up, sizeof(float3));
    render->GetCLKernel()->SetArgument(RenderKernelArgument_t::CAM_X, &camX, sizeof(float3));
    render->GetCLKernel()->SetArgument(RenderKernelArgument_t::CAM_EXTRA_DATA, &extra_data, sizeof(float4));
    render->GetCLKernel()->SetArgument(RenderKernelArgument_t::FRAME_COUNT, &m_FrameCount, sizeof(unsigned int));
    unsigned int seed = rand();
    render->GetCLKernel()->SetArgument(RenderKernelArgument_t::FRAME_SEED, &seed, sizeof(unsigned int));
    
    ++m_FrameCount;

}

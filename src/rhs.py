import torch

# ##################################################  LESGO    ###################################################

def padbcx(uinner):
    u_left = 2 * uinner[:, :, 0:1] - uinner[:, :, 1:2]  # 左边界外推
    u_right = 2 * uinner[:, :, -1:] - uinner[:, :, -2:-1]  # 右边界外推
    u_padded = torch.cat((u_left, uinner, u_right), dim=2)
    return u_padded


def padbcy(uinner):
    u_left = 2 * uinner[:, :, :, 0:1] - uinner[:, :, :, 1:2]  # 左边界外推
    u_right = 2 * uinner[:, :, :, -1:] - uinner[:, :, :, -2:-1]  # 右边界外推
    u_padded = torch.cat((u_left, uinner, u_right), dim=3)
    return u_padded
S = None

def LES2DFu(u1,v1,w1 ,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,vx,vy,wx,wy,delta,dx,dy,dx2,dy2):
    global S
    S = (2*(dudx(ux)/dx)**2 + 2*(dudy(vy)/dy)**2+(dudx(wx)/dx)**2+(dudy(wy)/dy)**2+(dudx(vx)/dx+dudy(uy)/dy)**2)**(1/2)

    return -u1 * dudx(ux)/dx - v1 * dudy(uy) / dy + (0.14 * delta)**2 * (S*(d2udx2(ux)/dx2 + d2udy2(uy)/dy2))
    # return -u1 * dudx(ux) / dx - v1 * dudy(uy) / dy


def LES2DFv(u1,v1,w1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,vx,vy,wx,wy,delta,dx,dy,dx2,dy2):

    return -u1*dudx(vx)/dx - v1*dudy(vy)/dy + (0.14 * delta)**2 * (S* (d2udx2(vx)/dx2 + d2udy2(vy)/dy2))
    # return -u1 * dudx(vx) / dx - v1 * dudy(vy) / dy


def LES2DFw(u1,v1,w1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,vx,vy,wx,wy,delta,dx,dy,dx2,dy2):

    return -u1*dudx(wx)/dx - v1*dudy(wy)/dy +(0.14*delta)**2 * (S*(d2udx2(wx)/dx2 + d2udy2(wy)/dy2))
    # return -u1*dudx(wx)/dx - v1*dudy(wy)/dy



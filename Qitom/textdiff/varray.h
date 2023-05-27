#pragma once
//#include <windows.h>

struct varray{
    unsigned char *ptr;
    int element_size;
    int length;
};

void varray_init(struct varray *p_pos,int element_size);
void varray_deinit(struct varray *p_pos);
void* varray_get(struct varray *p_pos,int i);

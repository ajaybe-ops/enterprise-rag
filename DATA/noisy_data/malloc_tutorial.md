A Malloc Tutorial

This tutorial explains how memory allocators work: the free list, splitting and
coalescing, and the difference between sbrk and mmap.

An allocator maintains a free list of free blocks. A malloc either splits a
free block that is larger than the request, or takes the block from the front of
the list. A free returns a block to the list and coalesces it with adjacent
free blocks to reduce fragmentation.

The classic allocator design is the Doug Lea allocator, first published in
1996. Its use of boundary tags allows coalescing with a free in constant time.

Fragmentation is the main practical problem. External fragmentation is free
memory that exists but is not contiguous; internal fragmentation is the unused
space inside an allocated block. Both reduce effective heap size without
any allocation failing outright.

sbrk extends the program break, a moving upper bound on the heap. mmap maps
anonymous pages, which are not subject to the program break and are returned to
the kernel on free. Large allocations generally go through mmap precisely
because of that property.

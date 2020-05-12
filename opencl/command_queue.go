package opencl

// #include "opencl.h"
import "C"

import (
	"unsafe"
)

type CommandQueue struct {
	commandQueue C.cl_command_queue
}

func createCommandQueue(context Context, device Device) (CommandQueue, error) {
	var errInt clError
	queue := C.clCreateCommandQueue(
		context.context,
		device.deviceID,
		0,
		(*C.cl_int)(&errInt),
	)
	if errInt != clSuccess {
		return CommandQueue{}, clErrorToError(errInt)
	}

	return CommandQueue{queue}, nil
}

func (c CommandQueue) EnqueueNDRangeKernel(kernel Kernel, workDim int, globalWorkSize []uint64) error {
	errInt := clError(C.clEnqueueNDRangeKernel(c.commandQueue,
		kernel.kernel,
		C.cl_uint(workDim),
		nil,
		(*C.size_t)(&globalWorkSize[0]),
		nil, 0, nil, nil))
	return clErrorToError(errInt)
}

func ToULong(x uint64) C.cl_ulong {
	return C.cl_ulong(x)
}

func (c CommandQueue) EnqueueWriteBuffer(buffer Buffer, blockingWrite bool, size uint64, dataPtr unsafe.Pointer) error {
	var bw C.cl_bool
	if blockingWrite {
		bw = C.CL_TRUE
	} else {
		bw = C.CL_FALSE
	}

	errInt := clError(C.clEnqueueWriteBuffer(c.commandQueue,
		buffer.buffer,
		bw,
		0,
		C.size_t(size),
		dataPtr,
		0, nil, nil))
	return clErrorToError(errInt)
}

func (c CommandQueue) EnqueueReadBuffer(buffer Buffer, blockingRead bool, size uint64, dataPtr unsafe.Pointer) error {
	var br C.cl_bool
	if blockingRead {
		br = C.CL_TRUE
	} else {
		br = C.CL_FALSE
	}

	errInt := clError(C.clEnqueueReadBuffer(c.commandQueue,
		buffer.buffer,
		br,
		0,
		C.size_t(size),
		dataPtr,
		0, nil, nil))
	return clErrorToError(errInt)
}

func (c CommandQueue) Release() {
	C.clReleaseCommandQueue(c.commandQueue)
}

func (c CommandQueue) Flush() {
	C.clFlush(c.commandQueue)
}

func (c CommandQueue) Finish() {
	C.clFinish(c.commandQueue)
}

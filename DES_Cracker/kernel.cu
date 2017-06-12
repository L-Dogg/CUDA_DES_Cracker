#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdint>
#include <ctime>

#define FIRSTBIT	0x8000000000000000
#define BLOCK_SIZE	1024
#define BLOCKS		2048
#define KNOWN_ZEROS	37
#define MSGLEN		1

__device__ int work = 1;

// Host matrices:
const int PC1[56] = {
	57, 49, 41, 33, 25, 17,  9,
	1, 58, 50, 42, 34, 26, 18,
	10,  2, 59, 51, 43, 35, 27,
	19, 11,  3, 60, 52, 44, 36,
	63, 55, 47, 39, 31, 23, 15,
	7, 62, 54, 46, 38, 30, 22,
	14,  6, 61, 53, 45, 37, 29,
	21, 13,  5, 28, 20, 12,  4
};
const int Rotations[16] = {
	1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1
};
const int PC2[48] = {
	14, 17, 11, 24,  1,  5,
	3, 28, 15,  6, 21, 10,
	23, 19, 12,  4, 26,  8,
	16,  7, 27, 20, 13,  2,
	41, 52, 31, 37, 47, 55,
	30, 40, 51, 45, 33, 48,
	44, 49, 39, 56, 34, 53,
	46, 42, 50, 36, 29, 32
};
const int InitialPermutation[64] = {
	58, 50, 42, 34, 26, 18, 10,  2,
	60, 52, 44, 36, 28, 20, 12,  4,
	62, 54, 46, 38, 30, 22, 14,  6,
	64, 56, 48, 40, 32, 24, 16,  8,
	57, 49, 41, 33, 25, 17,  9,  1,
	59, 51, 43, 35, 27, 19, 11,  3,
	61, 53, 45, 37, 29, 21, 13,  5,
	63, 55, 47, 39, 31, 23, 15,  7
};
const int FinalPermutation[64] = {
	40,  8, 48, 16, 56, 24, 64, 32,
	39,  7, 47, 15, 55, 23, 63, 31,
	38,  6, 46, 14, 54, 22, 62, 30,
	37,  5, 45, 13, 53, 21, 61, 29,
	36,  4, 44, 12, 52, 20, 60, 28,
	35,  3, 43, 11, 51, 19, 59, 27,
	34,  2, 42, 10, 50, 18, 58, 26,
	33,  1, 41,  9, 49, 17, 57, 25
};
const int DesExpansion[48] = {
	32,  1,  2,  3,  4,  5,  4,  5,
	6,  7,  8,  9,  8,  9, 10, 11,
	12, 13, 12, 13, 14, 15, 16, 17,
	16, 17, 18, 19, 20, 21, 20, 21,
	22, 23, 24, 25, 24, 25, 26, 27,
	28, 29, 28, 29, 30, 31, 32,  1
};
const int DesSbox[8][4][16] = {
	{
		{ 14,  4, 13,  1,  2, 15, 11,  8,  3, 10,  6, 12,  5,  9,  0,  7 },
		{ 0, 15,  7,  4, 14,  2, 13,  1, 10,  6, 12, 11,  9,  5,  3,  8 },
		{ 4,  1, 14,  8, 13,  6,  2, 11, 15, 12,  9,  7,  3, 10,  5,  0 },
		{ 15, 12,  8,  2,  4,  9,  1,  7,  5, 11,  3, 14, 10,  0,  6, 13 },
	},

	{
		{ 15,  1,  8, 14,  6, 11,  3,  4,  9,  7,  2, 13, 12,  0,  5, 10 },
		{ 3, 13,  4,  7, 15,  2,  8, 14, 12,  0,  1, 10,  6,  9, 11,  5 },
		{ 0, 14,  7, 11, 10,  4, 13,  1,  5,  8, 12,  6,  9,  3,  2, 15 },
		{ 13,  8, 10,  1,  3, 15,  4,  2, 11,  6,  7, 12,  0,  5, 14,  9 },
	},

	{
		{ 10,  0,  9, 14,  6,  3, 15,  5,  1, 13, 12,  7, 11,  4,  2,  8 },
		{ 13,  7,  0,  9,  3,  4,  6, 10,  2,  8,  5, 14, 12, 11, 15,  1 },
		{ 13,  6,  4,  9,  8, 15,  3,  0, 11,  1,  2, 12,  5, 10, 14,  7 },
		{ 1, 10, 13,  0,  6,  9,  8,  7,  4, 15, 14,  3, 11,  5,  2, 12 },
	},

	{
		{ 7, 13, 14,  3,  0,  6,  9, 10,  1,  2,  8,  5, 11, 12,  4, 15 },
		{ 13,  8, 11,  5,  6, 15,  0,  3,  4,  7,  2, 12,  1, 10, 14,  9 },
		{ 10,  6,  9,  0, 12, 11,  7, 13, 15,  1,  3, 14,  5,  2,  8,  4 },
		{ 3, 15,  0,  6, 10,  1, 13,  8,  9,  4,  5, 11, 12,  7,  2, 14 },
	},

	{
		{ 2, 12,  4,  1,  7, 10, 11,  6,  8,  5,  3, 15, 13,  0, 14,  9 },
		{ 14, 11,  2, 12,  4,  7, 13,  1,  5,  0, 15, 10,  3,  9,  8,  6 },
		{ 4,  2,  1, 11, 10, 13,  7,  8, 15,  9, 12,  5,  6,  3,  0, 14 },
		{ 11,  8, 12,  7,  1, 14,  2, 13,  6, 15,  0,  9, 10,  4,  5,  3 },
	},

	{
		{ 12,  1, 10, 15,  9,  2,  6,  8,  0, 13,  3,  4, 14,  7,  5, 11 },
		{ 10, 15,  4,  2,  7, 12,  9,  5,  6,  1, 13, 14,  0, 11,  3,  8 },
		{ 9, 14, 15,  5,  2,  8, 12,  3,  7,  0,  4, 10,  1, 13, 11,  6 },
		{ 4,  3,  2, 12,  9,  5, 15, 10, 11, 14,  1,  7,  6,  0,  8, 13 },
	},

	{
		{ 4, 11,  2, 14, 15,  0,  8, 13,  3, 12,  9,  7,  5, 10,  6,  1 },
		{ 13,  0, 11,  7,  4,  9,  1, 10, 14,  3,  5, 12,  2, 15,  8,  6 },
		{ 1,  4, 11, 13, 12,  3,  7, 14, 10, 15,  6,  8,  0,  5,  9,  2 },
		{ 6, 11, 13,  8,  1,  4, 10,  7,  9,  5,  0, 15, 14,  2,  3, 12 },
	},

	{
		{ 13,  2,  8,  4,  6, 15, 11,  1, 10,  9,  3, 14,  5,  0, 12,  7 },
		{ 1, 15, 13,  8, 10,  3,  7,  4, 12,  5,  6, 11,  0, 14,  9,  2 },
		{ 7, 11,  4,  1,  9, 12, 14,  2,  0,  6, 10, 13, 15,  3,  5,  8 },
		{ 2,  1, 14,  7,  4, 10,  8, 13, 15, 12,  9,  0,  3,  5,  6, 11 },
	},
};
const int Pbox[32] = {
	16,  7, 20, 21, 29, 12, 28, 17,
	1, 15, 23, 26,  5, 18, 31, 10,
	2,  8, 24, 14, 32, 27,  3,  9,
	19, 13, 30,  6, 22, 11,  4, 25
};

// Device matrices:
__constant__ int d_PC1[56] = {
	57, 49, 41, 33, 25, 17,  9,
	1, 58, 50, 42, 34, 26, 18,
	10,  2, 59, 51, 43, 35, 27,
	19, 11,  3, 60, 52, 44, 36,
	63, 55, 47, 39, 31, 23, 15,
	7, 62, 54, 46, 38, 30, 22,
	14,  6, 61, 53, 45, 37, 29,
	21, 13,  5, 28, 20, 12,  4
};
__constant__ int d_Rotations[16] = {
	1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1
};
__constant__ int d_PC2[48] = {
	14, 17, 11, 24,  1,  5,
	3, 28, 15,  6, 21, 10,
	23, 19, 12,  4, 26,  8,
	16,  7, 27, 20, 13,  2,
	41, 52, 31, 37, 47, 55,
	30, 40, 51, 45, 33, 48,
	44, 49, 39, 56, 34, 53,
	46, 42, 50, 36, 29, 32
};
__constant__ int d_InitialPermutation[64] = {
	58, 50, 42, 34, 26, 18, 10,  2,
	60, 52, 44, 36, 28, 20, 12,  4,
	62, 54, 46, 38, 30, 22, 14,  6,
	64, 56, 48, 40, 32, 24, 16,  8,
	57, 49, 41, 33, 25, 17,  9,  1,
	59, 51, 43, 35, 27, 19, 11,  3,
	61, 53, 45, 37, 29, 21, 13,  5,
	63, 55, 47, 39, 31, 23, 15,  7
};
__constant__ int d_FinalPermutation[64] = {
	40,  8, 48, 16, 56, 24, 64, 32,
	39,  7, 47, 15, 55, 23, 63, 31,
	38,  6, 46, 14, 54, 22, 62, 30,
	37,  5, 45, 13, 53, 21, 61, 29,
	36,  4, 44, 12, 52, 20, 60, 28,
	35,  3, 43, 11, 51, 19, 59, 27,
	34,  2, 42, 10, 50, 18, 58, 26,
	33,  1, 41,  9, 49, 17, 57, 25
};
__constant__ int d_DesExpansion[48] = {
	32,  1,  2,  3,  4,  5,  4,  5,
	6,  7,  8,  9,  8,  9, 10, 11,
	12, 13, 12, 13, 14, 15, 16, 17,
	16, 17, 18, 19, 20, 21, 20, 21,
	22, 23, 24, 25, 24, 25, 26, 27,
	28, 29, 28, 29, 30, 31, 32,  1
};
__constant__ int d_DesSbox[8][4][16] = {
	{
		{ 14,  4, 13,  1,  2, 15, 11,  8,  3, 10,  6, 12,  5,  9,  0,  7 },
		{ 0, 15,  7,  4, 14,  2, 13,  1, 10,  6, 12, 11,  9,  5,  3,  8 },
		{ 4,  1, 14,  8, 13,  6,  2, 11, 15, 12,  9,  7,  3, 10,  5,  0 },
		{ 15, 12,  8,  2,  4,  9,  1,  7,  5, 11,  3, 14, 10,  0,  6, 13 },
	},

	{
		{ 15,  1,  8, 14,  6, 11,  3,  4,  9,  7,  2, 13, 12,  0,  5, 10 },
		{ 3, 13,  4,  7, 15,  2,  8, 14, 12,  0,  1, 10,  6,  9, 11,  5 },
		{ 0, 14,  7, 11, 10,  4, 13,  1,  5,  8, 12,  6,  9,  3,  2, 15 },
		{ 13,  8, 10,  1,  3, 15,  4,  2, 11,  6,  7, 12,  0,  5, 14,  9 },
	},

	{
		{ 10,  0,  9, 14,  6,  3, 15,  5,  1, 13, 12,  7, 11,  4,  2,  8 },
		{ 13,  7,  0,  9,  3,  4,  6, 10,  2,  8,  5, 14, 12, 11, 15,  1 },
		{ 13,  6,  4,  9,  8, 15,  3,  0, 11,  1,  2, 12,  5, 10, 14,  7 },
		{ 1, 10, 13,  0,  6,  9,  8,  7,  4, 15, 14,  3, 11,  5,  2, 12 },
	},

	{
		{ 7, 13, 14,  3,  0,  6,  9, 10,  1,  2,  8,  5, 11, 12,  4, 15 },
		{ 13,  8, 11,  5,  6, 15,  0,  3,  4,  7,  2, 12,  1, 10, 14,  9 },
		{ 10,  6,  9,  0, 12, 11,  7, 13, 15,  1,  3, 14,  5,  2,  8,  4 },
		{ 3, 15,  0,  6, 10,  1, 13,  8,  9,  4,  5, 11, 12,  7,  2, 14 },
	},

	{
		{ 2, 12,  4,  1,  7, 10, 11,  6,  8,  5,  3, 15, 13,  0, 14,  9 },
		{ 14, 11,  2, 12,  4,  7, 13,  1,  5,  0, 15, 10,  3,  9,  8,  6 },
		{ 4,  2,  1, 11, 10, 13,  7,  8, 15,  9, 12,  5,  6,  3,  0, 14 },
		{ 11,  8, 12,  7,  1, 14,  2, 13,  6, 15,  0,  9, 10,  4,  5,  3 },
	},

	{
		{ 12,  1, 10, 15,  9,  2,  6,  8,  0, 13,  3,  4, 14,  7,  5, 11 },
		{ 10, 15,  4,  2,  7, 12,  9,  5,  6,  1, 13, 14,  0, 11,  3,  8 },
		{ 9, 14, 15,  5,  2,  8, 12,  3,  7,  0,  4, 10,  1, 13, 11,  6 },
		{ 4,  3,  2, 12,  9,  5, 15, 10, 11, 14,  1,  7,  6,  0,  8, 13 },
	},

	{
		{ 4, 11,  2, 14, 15,  0,  8, 13,  3, 12,  9,  7,  5, 10,  6,  1 },
		{ 13,  0, 11,  7,  4,  9,  1, 10, 14,  3,  5, 12,  2, 15,  8,  6 },
		{ 1,  4, 11, 13, 12,  3,  7, 14, 10, 15,  6,  8,  0,  5,  9,  2 },
		{ 6, 11, 13,  8,  1,  4, 10,  7,  9,  5,  0, 15, 14,  2,  3, 12 },
	},

	{
		{ 13,  2,  8,  4,  6, 15, 11,  1, 10,  9,  3, 14,  5,  0, 12,  7 },
		{ 1, 15, 13,  8, 10,  3,  7,  4, 12,  5,  6, 11,  0, 14,  9,  2 },
		{ 7, 11,  4,  1,  9, 12, 14,  2,  0,  6, 10, 13, 15,  3,  5,  8 },
		{ 2,  1, 14,  7,  4, 10,  8, 13, 15, 12,  9,  0,  3,  5,  6, 11 },
	},
};
__constant__ int d_Pbox[32] = {
	16,  7, 20, 21, 29, 12, 28, 17,
	1, 15, 23, 26,  5, 18, 31, 10,
	2,  8, 24, 14, 32, 27,  3,  9,
	19, 13, 30,  6, 22, 11,  4, 25
};

/*
 * Generates 16 keys (saved in generating order when encrypting or reverse order when decrypting) and saves them into keys[] array
 */
__device__ __host__ void generate_keys(uint64_t basekey, bool reverse, const int PC1[], const int PC2[], const int Rotations[], uint64_t keys[])
{
	uint64_t first = 0;
	for (int i = 0; i < 56; i++)
	{
		if (basekey & ((uint64_t)1 << (63 - (PC1[i] - 1))))
			first += ((uint64_t)1 << 63 - i);
	}

	uint64_t d[17];
	uint64_t c[17];

	const uint64_t mask = 0b0000000000000000000000000000111111111111111111111111111100000000;
	d[0] = (first & mask) << 28; //right half
	c[0] = ((first >> 28) & mask) << 28; //left half

	for (int i = 1; i <= 16; i++)
	{
		int shifts = Rotations[i - 1];
		c[i] = c[i - 1] << shifts;
		d[i] = d[i - 1] << shifts;

		if (c[i - 1] & (uint64_t)1 << 63)
			c[i] += (uint64_t)1 << 35 + shifts;
		if (shifts == 2)
			if (c[i - 1] & (uint64_t)1 << 62)
				c[i] += (uint64_t)1 << 36;

		if (d[i - 1] & (uint64_t)1 << 63)
			d[i] += (uint64_t)1 << 35 + shifts;
		if (shifts == 2)
			if (d[i - 1] & (uint64_t)1 << 62)
				d[i] += (uint64_t)1 << 36;


		keys[i] = c[i] | (d[i] >> 28);
		uint64_t tmp = 0;
		for (int j = 0; j < 48; j++)
		{
			if (keys[i] & ((uint64_t)1 << (63 - (PC2[j] - 1))))
				tmp += ((uint64_t)1 << 63 - j);
		}
		keys[i] = tmp;
	}

	if (reverse)
	{
		for (int i = 1; i <= 8; i++)
		{
			uint64_t tmp = keys[i];
			keys[i] = keys[17 - i];
			keys[17 - i] = tmp;
		}
	}
}

/*
 * Prints all v bits from start to end given by arguments
 */
__device__ __host__ void printbits(uint64_t v, int start = 0, int end = 64)
{
	for (int ii = start; ii < end; ii++)
	{
		if (((v << ii) & FIRSTBIT) == (uint64_t)0)
			printf("0");
		else
			printf("1");
	}
	printf("\n");
}

/*
 * Permutates block using initial permutation matrix or final permutation matrix (determined by the second argument)
 */
__device__ __host__ uint64_t permutate_block(uint64_t block, bool initial, const int InitialPermutation[], const int FinalPermutation[])
{
	uint64_t permutation = 0;
	for (int i = 0; i < 64; i++)
	{
		if (initial)
		{
			if (block & ((uint64_t)1 << (63 - (InitialPermutation[i] - 1))))
				permutation += ((uint64_t)1 << 63 - i);
		}
		else if (block & ((uint64_t)1 << (63 - (FinalPermutation[i] - 1))))
			permutation += ((uint64_t)1 << 63 - i);
	}

	return permutation;
}

/*
 * Expands block using Expansion matrix given by the argument
 */
__device__ __host__ uint64_t expand(uint64_t val, const int DesExpansion[])
{
	uint64_t res = 0;
	for (int i = 0; i < 48; i++)
	{
		if (val & ((uint64_t)1 << (63 - (DesExpansion[i] - 1))))
			res += ((uint64_t)1 << 63 - i);
	}
	return res;
}

__device__ __host__ uint64_t calculate_sboxes(uint64_t val, const int DesSbox[8][4][16])
{
	uint64_t mask = 0b1111110000000000000000000000000000000000000000000000000000000000;
	uint64_t middle_bits = 0b0000000000000000000000000000000000000000000000000000000000011110;
	uint64_t ret = 0;
	for (int i = 0; i < 8; i++)
	{
		uint64_t current = (val & (mask >> (6 * i))) >> (64 - 6 * (i + 1));
		int column = (current & middle_bits) >> 1;
		int row = ((current & (1 << 5)) >> 4) + (current & 1);
		uint64_t val = DesSbox[i][row][column];
		ret += val << (60 - 4 * i);
	}
	return ret;
}

/*
 * Calculates block encryption/decryption
 */
__device__ __host__ uint64_t jechanka(uint64_t permutated, uint64_t keys[], const int PC1[], const int Rotations[], const int PC2[], 
									  const int InitialPermutation[], const int FinalPermutation[], const int DesExpansion[], 
									  const int Sbox[8][4][16], const int Pbox[])
{
	uint64_t l[17], r[17];
	uint64_t mask = 0b1111111111111111111111111111111100000000000000000000000000000000;
	l[0] = permutated & mask;
	r[0] = (permutated << 32) & mask;

	for (int i = 1; i <= 16; i++)
	{
		l[i] = r[i - 1];
		uint64_t v = calculate_sboxes(keys[i] ^ expand(r[i - 1], DesExpansion), Sbox);
		uint64_t res = 0;
		for (int j = 0; j < 32; j++)
		{
			if (v & ((uint64_t)1 << (63 - (Pbox[j] - 1))))
				res += ((uint64_t)1 << 63 - j);
		}
		r[i] = l[i - 1] ^ res;
	}

	return permutate_block(r[16] + (l[16] >> 32), false, InitialPermutation, FinalPermutation);
}

/*
 * Main DES function - decrypts or encrypts whole message (length given by MSGLEN constant)
 */
__device__ __host__ void DES(uint64_t encryptedMessage[], uint64_t decryptedMessage[], uint64_t key, const int PC1[], const int Rotations[],
							 const int PC2[], const int InitialPermutation[], const int FinalPermutation[], const int DesExpansion[],
							 const int Sbox[8][4][16], const int Pbox[], bool encrypt)
{
	uint64_t keys[17];
	generate_keys(key, !encrypt, PC1, PC2, Rotations, keys);
	for (int i = 0; i < MSGLEN; i++)
	{
		if (encrypt)
			encryptedMessage[i] = jechanka(permutate_block(decryptedMessage[i], true, InitialPermutation, FinalPermutation), keys, PC1, Rotations,
				PC2, InitialPermutation, FinalPermutation, DesExpansion, Sbox, Pbox);
		else
			decryptedMessage[i] = jechanka(permutate_block(encryptedMessage[i], true, InitialPermutation, FinalPermutation), keys, PC1, Rotations,
				PC2, InitialPermutation, FinalPermutation, DesExpansion, Sbox, Pbox);

	}
}

__global__ void worker_thread(const uint64_t message[], uint64_t encrypted[], uint64_t decrypted[], int known_zeros)
{
	uint64_t threadId = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	uint64_t mask = 0b0000000000000000000000000000000000000000000000000000000001111111;
	uint64_t suffix = ((threadId & (mask << 14)) << 3) | ((threadId & (mask << 7)) << 2) | ((threadId & mask) << 1);
	uint64_t current_key = 0;
	uint64_t max = (uint64_t)1 << (35 - (known_zeros - known_zeros / 8));
	uint64_t current_message[MSGLEN];
	bool go;

	for (uint64_t i = 0; i < max && work == 1; i++)
	{
		current_key = (((i & (mask << 28)) << 5) | (((i & (mask << 21)) << 4) | ((i & (mask << 14)) << 3) | ((i & (mask << 7)) << 2) | ((i & mask) << 1)) << 24) | suffix;
		DES(encrypted, current_message, current_key, d_PC1, d_Rotations, d_PC2, d_InitialPermutation, d_FinalPermutation, d_DesExpansion, d_DesSbox, d_Pbox, false);

		go = true;
		for (int j = 0; j < MSGLEN; j++) 
		{
			if (current_message[j] != message[j]) 
			{
				go = false;
				break;
			}
		}

		if (go) 
		{
			for (int j = 0; j < MSGLEN; j++) 
			{
				decrypted[j] = current_message[j];
			}
			work = 0;
		}
	}
}

/*
 * Decrypts message using brute-force method and CUDA
 */
cudaError_t CudaDES(uint64_t plaintext[], uint64_t encrypted[], uint64_t decrypted[], uint64_t key)
{
	cudaError_t cudaStatus;

	uint64_t *d_plain, *d_enc, *d_dec;

	cudaStatus = cudaMalloc(&d_plain, MSGLEN * sizeof(uint64_t));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_enc, MSGLEN * sizeof(uint64_t));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_dec, MSGLEN * sizeof(uint64_t));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_plain, plaintext, MSGLEN * sizeof(uint64_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_enc, encrypted, MSGLEN * sizeof(uint64_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	worker_thread << < BLOCKS, BLOCK_SIZE >> > (d_plain, d_enc, d_dec, KNOWN_ZEROS);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "worker thread failed!");
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(decrypted, d_dec, MSGLEN * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(d_plain);
	cudaFree(d_enc);
	cudaFree(d_dec);

	return cudaStatus;
}

/*
 * Prints message in binary format
 */
void printmsg(uint64_t msg[])
{
	for (int i = 0; i < MSGLEN; i++)
		printbits(msg[i]);
}

/*
 * Compares original message with message decrypted with CUDA/CPU
 */
bool proper_decipher(uint64_t msg[], uint64_t decrypted[])
{
	for (int i = 0; i < MSGLEN; i++)
	{
		if (msg[i] != decrypted[i])
		{
			return false;
		}
	}
	return true;
}

void cpuDES(uint64_t plaintext[], uint64_t encrypted[], uint64_t decrypted[], uint64_t key, int known_zeros)
{
	int cpu_work = 1;
	for (uint64_t threadId = 0; threadId < (uint64_t)2 << 21 && cpu_work == 1; threadId++)
	{
		uint64_t mask = 0b0000000000000000000000000000000000000000000000000000000001111111;
		uint64_t suffix = ((threadId & (mask << 14)) << 3) | ((threadId & (mask << 7)) << 2) | ((threadId & mask) << 1);
		uint64_t current_key = 0;
		uint64_t max = (uint64_t)1 << (35 - (known_zeros - known_zeros / 8));
		uint64_t current_message[MSGLEN];
		bool go;

		for (uint64_t i = 0; i < max && cpu_work == 1; i++)
		{
			current_key = (((i & (mask << 28)) << 5) | (((i & (mask << 21)) << 4) | ((i & (mask << 14)) << 3) | ((i & (mask << 7)) << 2) | ((i & mask) << 1)) << 24) | suffix;
			DES(encrypted, current_message, current_key, PC1, Rotations, PC2, InitialPermutation, FinalPermutation, DesExpansion, DesSbox, Pbox, false);

			go = true;
			for (int j = 0; j < MSGLEN; j++)
			{
				if (current_message[j] != plaintext[j])
				{
					go = false;
					break;
				}
			}

			if (go)
			{
				for (int j = 0; j < MSGLEN; j++)
				{
					decrypted[j] = current_message[j];
				}
				cpu_work = 0;
			}
		}
	}
}

int main()
{
	uint64_t key = 2;
	uint64_t msg[1] = { 0b00000000000100100011010001010110011110001001101010111100110111101111 };
	uint64_t decrypted[1];
	uint64_t encrypted[1];
	bool success;
	printf("Plain text:\n");
	printmsg(msg);

	DES(encrypted, msg, key, PC1, Rotations, PC2, InitialPermutation, FinalPermutation, DesExpansion, DesSbox, Pbox, true);

	printf("Encrypted:\n");
	printmsg(encrypted);

	clock_t begin, end;
	double elapsed_secs;

/* --------------------------------------- CUDA ----------------------------------------------------- */

	//begin = clock();
	//printf("Starting GPU DES cracking for %d known leading zeros...\n", KNOWN_ZEROS);
	//cudaError_t cudaStatus = CudaDES(msg, encrypted, decrypted, key);
	//end = clock();
	//elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	//printf("Finished GPU DES cracking for known leading zeros count = %d.\nTime elapsed: %fs.\n", KNOWN_ZEROS, elapsed_secs);
	//if (cudaStatus != cudaSuccess)
	//{
	//	printf("Cos sie, cos sie popsulo...\n");
	//}
	//else
	//{
	//	printf("Decrypted:\n");
	//	printmsg(decrypted);
	//	success = proper_decipher(msg, decrypted);
	//	printf(success ? "SUCCESS\n" : "FAILURE\n");
	//}

/* --------------------------------------- CPU ----------------------------------------------------- */
	
	begin = clock();
	printf("Starting CPU DES cracking for %d known leading zeros...\n", KNOWN_ZEROS);
	cpuDES(msg, encrypted, decrypted, key, KNOWN_ZEROS);
	end = clock();
	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	printf("Finished CPU DES cracking for known leading zeros count = %d.\nTime elapsed: %fs.\n", KNOWN_ZEROS, elapsed_secs);
	
	printf("Decrypted:\n");
	printmsg(decrypted);
	success = proper_decipher(msg, decrypted);
	printf(success ? "SUCCESS\n" : "FAILURE\n");
	
/* --------------------------------------- DEBUG ----------------------------------------------------- */

	DES(encrypted, decrypted, key, PC1, Rotations, PC2, InitialPermutation, FinalPermutation, DesExpansion, DesSbox, Pbox, false);
	printf("Decrypted with proper key:\n");
	printmsg(decrypted);
	success = proper_decipher(msg, decrypted);
	printf(success ? "SUCCESS\n" : "FAILURE\n");

	printf("Press any key to exit program\n");
	getchar();
	return 0;
}
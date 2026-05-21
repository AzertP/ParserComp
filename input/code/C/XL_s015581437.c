#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ORDER_LENGTH 100
#define NUM_DICE_FACE    6
#define END_LOOP         0
#define NUM_DICE         2

#define TRUE             1
#define FALSE            0

typedef unsigned char   U1;
typedef unsigned short  U2;
typedef unsigned long   U4;
typedef signed   char   S1;
typedef signed   short  S2;
typedef signed   long   S4;

struct Dice {
	U1 u1Top;
	U1 u1South;
	U1 u1East;
	U1 u1North;
	U1 u1West;
	U1 u1Bottom;
};

/********************************************************************************/
/* 関数名   | inputDiceInfo                                                     */
/* 戻り値   | 無し                                                              */
/* 備考     | 無し                                                              */
/********************************************************************************/
void inputDiceInfo(U1 *u1DiceInfo)
{
	int intInput;
	U1 u1Cnt;
	for (u1Cnt = 0; u1Cnt < NUM_DICE_FACE; u1Cnt++)
	{
		scanf("%d", &intInput);
		u1DiceInfo[u1Cnt] = intInput;
	}
}

/********************************************************************************/
/* 関数名   | inputDiceInfo                                                     */
/* 戻り値   | 無し                                                              */
/* 備考     | 無し                                                              */
/********************************************************************************/
void inputCommand(char *chCommand)
{
	scanf("%s", chCommand);
}

/********************************************************************************/
/* 関数名   | initData                                                          */
/* 戻り値   | 無し                                                              */
/* 備考     | 無し                                                              */
/********************************************************************************/
void initData(struct Dice *stDice, U1 *u1DiceInfo)
{
	U1 u1Cnt;
	U1 *u1TmpDiceNum;

	u1TmpDiceNum = u1DiceInfo;

	stDice->u1Top = (U1)u1DiceInfo[0];
	stDice->u1South = (U1)u1DiceInfo[1];
	stDice->u1East = (U1)u1DiceInfo[2];
	stDice->u1West = (U1)u1DiceInfo[3];
	stDice->u1North = (U1)u1DiceInfo[4];
	stDice->u1Bottom = (U1)u1DiceInfo[5];
}

/********************************************************************************/
/* 関数名   | cpyDice                                                           */
/* 戻り値   | 無し                                                              */
/* 備考     | 無し                                                              */
/********************************************************************************/
void cpyDice(struct Dice *stCopyDice, struct Dice *stPasteDice)
{
	stPasteDice->u1Top    = stCopyDice->u1Top;
	stPasteDice->u1South  = stCopyDice->u1South;
	stPasteDice->u1East   = stCopyDice->u1East;
	stPasteDice->u1North  = stCopyDice->u1North;
	stPasteDice->u1West   = stCopyDice->u1West;
	stPasteDice->u1Bottom = stCopyDice->u1Bottom;
}

/********************************************************************************/
/* 関数名   | chngDice                                                          */
/* 戻り値   | 無し                                                              */
/* 備考     | 無し                                                              */
/********************************************************************************/
void chngDice(struct Dice *stDice, char chCommand)
{
	U1 u1TmpVal;
	if (chCommand == 'N')
	{
		u1TmpVal = stDice->u1Top;
		stDice->u1Top = stDice->u1South;
		stDice->u1South = stDice->u1Bottom;
		stDice->u1Bottom = stDice->u1North;
		stDice->u1North = u1TmpVal;
	}
	else if (chCommand == 'S')
	{
		u1TmpVal = stDice->u1Top;
		stDice->u1Top = stDice->u1North;
		stDice->u1North = stDice->u1Bottom;
		stDice->u1Bottom = stDice->u1South;
		stDice->u1South = u1TmpVal;
	}
	else if (chCommand == 'W')
	{
		u1TmpVal = stDice->u1Top;
		stDice->u1Top = stDice->u1East;
		stDice->u1East = stDice->u1Bottom;
		stDice->u1Bottom = stDice->u1West;
		stDice->u1West = u1TmpVal;
	}
	else if (chCommand == 'E')
	{
		u1TmpVal = stDice->u1Top;
		stDice->u1Top = stDice->u1West;
		stDice->u1West = stDice->u1Bottom;
		stDice->u1Bottom = stDice->u1East;
		stDice->u1East = u1TmpVal;
	}
	else if (chCommand == 'R')
	{
		u1TmpVal = stDice->u1South;
		stDice->u1South = stDice->u1East;
		stDice->u1East = stDice->u1North;
		stDice->u1North = stDice->u1West;
		stDice->u1West = u1TmpVal;
	}
	else
	{
		;
	}
}

/********************************************************************************/
/* 関数名   | judgeDiceExceptTop                                                */
/* 戻り値   | TRUE:サイコロの上面以外が一致                                     */
/*          | FALSE:サイコロの上面以外が不一致                                  */
/* 備考     | サイコロの上面以外を比較する                                      */
/********************************************************************************/
U1 judgeDiceExceptTop(struct Dice *stDice1, struct Dice *stDice2)
{
	U1 u1FaceCnt;
	U1 u1Ret;

	u1Ret = FALSE;
	/* bottomの比較 */
	if (stDice1->u1Bottom == stDice2->u1Bottom)
	{
		/* topとbottom以外の比較 */
		for (u1FaceCnt = 0; u1FaceCnt < 4; u1FaceCnt++)
		{
			if ((stDice1->u1South == stDice2->u1South) &&
				(stDice1->u1East == stDice2->u1East) &&
				(stDice1->u1North == stDice2->u1North) &&
				(stDice1->u1West == stDice2->u1West))
			{
				u1Ret = TRUE;
				break;
			}
			else
			{
				chngDice(stDice2, 'R');
			}
		}
	}
	return u1Ret;
}

/********************************************************************************/
/* 関数名   | judgeSameDices                                                    */
/* 戻り値   | TRUE:サイコロの全面が一致                                         */
/*          | FALSE:サイコロの全面が不一致                                      */
/* 備考     | サイコロの全面を比較する                                          */
/********************************************************************************/
U1 judgeSameDices(struct Dice stDice[])
{
	struct Dice stTmpDice[NUM_DICE];
	U1 u1DiceCnt;
	U1 u1FaceCnt;
	U1 u1Ret;

	u1Ret = FALSE;
	for (u1DiceCnt = 0; u1DiceCnt < NUM_DICE; u1DiceCnt++)
	{
        cpyDice(&stDice[u1DiceCnt],&stTmpDice[u1DiceCnt]);
	}

	/* Topの比較 */
	for (u1DiceCnt = 1; u1DiceCnt < NUM_DICE; u1DiceCnt++)
	{
        /* Top-Topの比較 */
		if (stTmpDice[0].u1Top == stTmpDice[u1DiceCnt].u1Top)
		{
			u1Ret = judgeDiceExceptTop(&stTmpDice[0], &stTmpDice[u1DiceCnt]);
			if (u1Ret == TRUE)
			{
				break;
			}
			else
			{
                cpyDice(&stDice[u1DiceCnt],&stTmpDice[u1DiceCnt]);
			}
		}
		if (stTmpDice[0].u1Top == stTmpDice[u1DiceCnt].u1South)
		{
			chngDice(&stTmpDice[u1DiceCnt], 'N');
			u1Ret = judgeDiceExceptTop(&stTmpDice[0], &stTmpDice[u1DiceCnt]);
			if (u1Ret == TRUE)
			{
				break;
			}
			else
			{
                cpyDice(&stDice[u1DiceCnt],&stTmpDice[u1DiceCnt]);
			}
		}
		if (stTmpDice[0].u1Top == stTmpDice[u1DiceCnt].u1East)
		{
			chngDice(&stTmpDice[u1DiceCnt], 'W');
			u1Ret = judgeDiceExceptTop(&stTmpDice[0], &stTmpDice[u1DiceCnt]);
			if (u1Ret == TRUE)
			{
				break;
			}
			else
			{
                cpyDice(&stDice[u1DiceCnt],&stTmpDice[u1DiceCnt]);
			}
		}
		if (stTmpDice[0].u1Top == stTmpDice[u1DiceCnt].u1North)
		{
			chngDice(&stTmpDice[u1DiceCnt], 'S');
			u1Ret = judgeDiceExceptTop(&stTmpDice[0], &stTmpDice[u1DiceCnt]);
			if (u1Ret == TRUE)
			{
				break;
			}
			else
			{
                cpyDice(&stDice[u1DiceCnt],&stTmpDice[u1DiceCnt]);
			}
		}
		if (stTmpDice[0].u1Top == stTmpDice[u1DiceCnt].u1West)
		{
			chngDice(&stTmpDice[u1DiceCnt], 'E');
			u1Ret = judgeDiceExceptTop(&stTmpDice[0], &stTmpDice[u1DiceCnt]);
			if (u1Ret == TRUE)
			{
				break;
			}
			else
			{
                cpyDice(&stDice[u1DiceCnt],&stTmpDice[u1DiceCnt]);
			}
		}
		if (stTmpDice[0].u1Top == stTmpDice[u1DiceCnt].u1Bottom)
		{
			chngDice(&stTmpDice[u1DiceCnt], 'N');
			chngDice(&stTmpDice[u1DiceCnt], 'N');
			u1Ret = judgeDiceExceptTop(&stTmpDice[0], &stTmpDice[u1DiceCnt]);
			if (u1Ret == TRUE)
			{
				break;
			}
			else
			{
                cpyDice(&stDice[u1DiceCnt],&stTmpDice[u1DiceCnt]);
			}
		}
		else
		{
			break;
		}
	}

	return u1Ret;
}

/********************************************************************************/
/* 関数名   | outputJudgeResult                                                 */
/* 戻り値   | 無し                                                              */
/* 備考     | 無し                                                              */
/********************************************************************************/
void outputJudgeResult(U1 u1Result)
{
	if (u1Result == TRUE)
	{
		printf("Yes\n");
	}
	else
	{
		printf("No\n");
	}
}

/********************************************************************************/
/* 関数名   | main                                                              */
/* 引数     | 無し                                                              */
/* 戻り値   | 無し                                                              */
/* 備考     | 無し                                                              */
/********************************************************************************/
int main(void)
{
	struct Dice stDice[NUM_DICE];
	U1 u1DiceInfo[NUM_DICE][NUM_DICE_FACE];
	U1 u1DiceCnt;
	U1 u1JudgeDice;

	/* 入力部 */
	for (u1DiceCnt = 0; u1DiceCnt < NUM_DICE; u1DiceCnt++)
	{
		inputDiceInfo(&u1DiceInfo[u1DiceCnt][0]);
		initData(&stDice[u1DiceCnt], &u1DiceInfo[u1DiceCnt][0]);
	}

	/* 制御部 */
	u1JudgeDice = judgeSameDices(&stDice);

	/* 出力部 */
	outputJudgeResult(u1JudgeDice);
}


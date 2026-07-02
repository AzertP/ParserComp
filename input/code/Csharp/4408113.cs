using System;

public class ITP1_11_C{
    public static void Main(){
        var values = Console.ReadLine().Split(' ');
        Dice dice1 = new Dice
        {
            Top = values[0],
            South = values[1],
            East = values[2],
            West = values[3],
            North = values[4],
            Bottom = values[5]
        };
        
        values = Console.ReadLine().Split(' ');
        Dice dice2 = new Dice
        {
            Top = values[0],
            South = values[1],
            East = values[2],
            West = values[3],
            North = values[4],
            Bottom = values[5]
        };
        
        if (dice1.IsIdenticalTo(dice2))
        {
            Console.WriteLine("Yes");
        }
        else
        {
            Console.WriteLine("No");
        }
        
    }
    
    class Dice
    {
        public string Top { get; set; }
        public string Bottom { get; set; }
        public string North { get; set; }
        public string South { get; set; }
        public string East { get; set; }
        public string West { get; set; }
        
        public void　RollToNorth()
        {
            string tmp = this.North;
            this.North = this.Top;
            this.Top = this.South;
            this.South = this.Bottom;
            this.Bottom = tmp;
        }
        
        public void　RollToSouth()
        {
            string tmp = this.South;
            this.South = this.Top;
            this.Top = this.North;
            this.North = this.Bottom;
            this.Bottom = tmp;
        }
        
        public void　RollToEast()
        {
            string tmp = this.East;
            this.East = this.Top;
            this.Top = this.West;
            this.West = this.Bottom;
            this.Bottom = tmp;
        }
        
        public void　RollToWest()
        {
            string tmp = this.West;
            this.West = this.Top;
            this.Top = this.East;
            this.East = this.Bottom;
            this.Bottom = tmp;
        }
        
        public void RotateToRight()
        {
            string tmp = this.North;
            this.North = this.West;
            this.West = this.South;
            this.South = this.East;
            this.East = tmp;
        }
        
        public void RotateToLeft()
        {
            string tmp = this.North;
            this.North = this.East;
            this.East = this.South;
            this.South = this.West;
            this.West = tmp;
        }
        
        public bool IsPerfectMatchWith(Dice dice)
        {
            return (this.Top == dice.Top && this.Bottom == dice.Bottom &&
                this.North == dice.North && this.East == dice.East &&
                this.South == dice.South && this.West == dice.West);
        }
        
        public bool IsIdenticalTo(Dice dice)
        {
            for (var i = 0; i < 6; i++)
            {
                if (i % 2 == 0)
                {
                    dice.RollToNorth();
                }
                else
                {
                    dice.RollToWest();
                }
                
                for (var j = 0; j < 4; j++)
                {
                    dice.RotateToRight();
                    if (this.IsPerfectMatchWith(dice))
                    {
                        return true;
                    }
                }
            }
            
            return false;
        }
    }
}

using System;

public class ITP1_11_A{
    public static void Main(){
        var values = Console.ReadLine().Split(' ');
        
        Dice dice = new Dice
        {
            Top = values[0],
            South = values[1],
            East = values[2],
            West = values[3],
            North = values[4],
            Bottom = values[5]
        };
        
        foreach (char cmd in Console.ReadLine())
        {
            switch (cmd)
            {
                case 'E':
                    dice.RollToEast();
                    break;
                case 'N':
                    dice.RollToNorth();
                    break;
                case 'S':
                    dice.RollToSouth();
                    break;
                default:
                    dice.RollToWest();
                    break;
            }
        }
        
        Console.WriteLine(dice.Top);
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
    }
}


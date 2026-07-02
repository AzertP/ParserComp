using System;

public class ITP1_11_B{
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
        
        var q = int.Parse(Console.ReadLine());
        
        
        for (var i = 0; i < q; i++)
        {
            var question = Console.ReadLine().Split();
            var top = question[0];
            var front = question[1];
            
            while (dice.Top != top)
            {
                dice.RollToNorth();
                if (dice.Top == top)
                {
                    break;
                }
                dice.RollToWest();
                if (dice.Top == top)
                {
                    break;
                }
            }
            
            while (dice.South != front)
            {
                dice.RotateToRight();
            }
                
            Console.WriteLine(dice.East);
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
    }
}

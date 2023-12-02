#include <iostream>
#include "Game.h"
#include "Common.h"

int main(int argc, char *argv[])
{
    StartupOptions options;
    parseArgs(argc, argv, options);
    Game game = Game(options);
    game.Run();
    saveToFile(options.outputFile, game.flock);
    return 0;
}

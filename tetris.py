#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The classic Tetris developed using PyGame.
    Copyright (C) 2018 Recursos Python - recursospython.com.
"""

from collections import OrderedDict
import random
from typing_extensions import Self

from pygame import Rect
import pygame
import numpy as np


WINDOW_WIDTH, WINDOW_HEIGHT = 500, 601
GRID_WIDTH, GRID_HEIGHT = 300, 600
TILE_SIZE = 30


def remove_empty_columns(arr, _x_offset=0, _keep_counting=True):
    """
    Remove empty columns from arr (i.e. those filled with zeros).
    The return value is (new_arr, x_offset), where x_offset is how
    much the x coordinate needs to be increased in order to maintain
    the block's original position.
    """
    for colid, col in enumerate(arr.T):
        if col.max() == 0:
            if _keep_counting:
                _x_offset += 1
            # Remove the current column and try again.
            arr, _x_offset = remove_empty_columns(
                np.delete(arr, colid, 1), _x_offset, _keep_counting)
            break
        else:
            _keep_counting = False
    return arr, _x_offset


class BottomReached(Exception):
    pass


class TopReached(Exception):
    pass


class Block(pygame.sprite.Sprite):
    
    @staticmethod
    def collide(block, group):
        """
        Check if the specified block collides with some other block
        in the group.
        """
        for other_block in group:
            # Ignore the current block which will always collide with itself.
            if block == other_block:
                continue
            if pygame.sprite.collide_mask(block, other_block) is not None:
                return True
        return False
    
    def __init__(self):
        super().__init__()
        # Get a random color.
        self.color = random.choice((
            (200, 200, 200),
            (215, 133, 133),
            (30, 145, 255),
            (0, 170, 0),
            (180, 0, 140),
            (200, 200, 0)
        ))
        self.current = True
        self.struct = np.array(self.struct)
        # Initial random rotation and flip.
        if random.randint(0, 1):
            self.struct = np.rot90(self.struct)
        if random.randint(0, 1):
            # Flip in the X axis.
            self.struct = np.flip(self.struct, 0)
        self._draw()
    
    def _draw(self, x=4, y=0):
        width = len(self.struct[0]) * TILE_SIZE
        height = len(self.struct) * TILE_SIZE
        self.image = pygame.surface.Surface([width, height])
        self.image.set_colorkey((0, 0, 0))
        # Position and size
        self.rect = Rect(0, 0, width, height)
        self.x = x
        self.y = y
        for y, row in enumerate(self.struct):
            for x, col in enumerate(row):
                if col:
                    pygame.draw.rect(
                        self.image,
                        self.color,
                        Rect(x*TILE_SIZE + 1, y*TILE_SIZE + 1,
                             TILE_SIZE - 2, TILE_SIZE - 2)
                    )
        self._create_mask()
    
    def redraw(self):
        self._draw(self.x, self.y)
    
    def _create_mask(self):
        """
        Create the mask attribute from the main surface.
        The mask is required to check collisions. This should be called
        after the surface is created or update.
        """
        self.mask = pygame.mask.from_surface(self.image)
    
    def initial_draw(self):
        raise NotImplementedError
    
    @property
    def group(self):
        return self.groups()[0]
    
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        self._x = value
        self.rect.left = value*TILE_SIZE
    
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        self._y = value
        self.rect.top = value*TILE_SIZE
    
    def move_left(self, group):
        self.x -= 1
        # Check if we reached the left margin.
        if self.x < 0 or Block.collide(self, group):
            self.x += 1
    
    def move_right(self, group):
        self.x += 1
        # Check if we reached the right margin or collided with another
        # block.
        if self.rect.right > GRID_WIDTH or Block.collide(self, group):
            # Rollback.
            self.x -= 1
    
    def move_down(self, group):
        self.y += 1
        # Check if the block reached the bottom or collided with 
        # another one.
        if self.rect.bottom > GRID_HEIGHT or Block.collide(self, group):
            # Rollback to the previous position.
            self.y -= 1
            self.current = False
            raise BottomReached
    
    def rotate(self, group):
        self.image = pygame.transform.rotate(self.image, 90)
        # Once rotated we need to update the size and position.
        self.rect.width = self.image.get_width()
        self.rect.height = self.image.get_height()
        self._create_mask()
        # Check the new position doesn't exceed the limits or collide
        # with other blocks and adjust it if necessary.
        while self.rect.right > GRID_WIDTH:
            self.x -= 1
        while self.rect.left < 0:
            self.x += 1
        while self.rect.bottom > GRID_HEIGHT:
            self.y -= 1
        while True:
            if not Block.collide(self, group):
                break
            self.y -= 1
        self.struct = np.rot90(self.struct)
    
    def update(self):
        if self.current:
            self.move_down()


class SquareBlock(Block):
    struct = (
        (1, 1),
        (1, 1)
    )

    def desc(self):
        return "SquareBlock"


class TBlock(Block):
    struct = (
        (1, 1, 1),
        (0, 1, 0)
    )

    def desc(self):
        return "TBlock"
    
class LineBlock(Block):
    struct = (
        (1,),
        (1,),
        (1,),
        (1,)
    )
    def desc(self):
        return "LineBlock"

class LBlock(Block):
    struct = (
        (1, 1),
        (1, 0),
        (1, 0),
    )
    def desc(self):
        return "LBlock"

class ZBlock(Block):
    struct = (
        (0, 1),
        (1, 1),
        (1, 0),
    )
    def desc(self):
        return "ZBlock"

class BlocksGroup(pygame.sprite.OrderedUpdates):

    def process_current_state(self):
        # Recieives current state to predict best move
        # Check if there's a new piece in the board
        if self.current_block.y == 0:
            # Analyze all possible moves and score them
            possible_moves_scored = self.score_all_possible_moves()
            # Pick best one
            highest_score = possible_moves_scored[0]["score"]
            self.current_selected_move = possible_moves_scored[0]
            for move in possible_moves_scored:
                if move["score"] > highest_score:
                    highest_score = move["score"]
                    # Use that as the next general trajectory
                    self.current_selected_move = move
        #print("rotation_mode:",self.current_selected_move["rotation_mode"],", i:",self.current_selected_move["i"],", j:",self.current_selected_move["j"],", score:",self.current_selected_move["score"])                
        # Pick move in movement frame from current trajectory

    def score_all_possible_moves(self):
        # Analyse all possible moves
        possible_moves = []
        # First check all possible rotations
        
        j_position = self.current_block.x
        for rotation_mode in range(4):
            # Rotate to this rotation
            self.rotate_current_block()
            rotated_shape = self.current_block
            # Now check all possible positions for this rotation
            for j_position in range(0, len(self.grid[0]) - \
                                    rotated_shape.struct.shape[1] + 1):
                i_position = self.current_block.y
                while(not self.blocks_below(i_position,j_position)):
                    # Make the block go down until it stops
                    i_position += 1
                # Create possible grid with this new position
                possible_final_state = self.create_final_state(i_position,j_position)
                # Score this position
                possible_final_state_score = self.score_state(possible_final_state)
                possible_moves.append({ "rotation_mode": rotation_mode, 
                                        "i": i_position, "j": j_position,
                                        "score": possible_final_state_score,
                                        "state": possible_final_state })
        return possible_moves 

#grid, shape, i_position, j_position
    def create_final_state(self,i_position,j_position):
        # Add new possible piece position to grid to score this state
        new_full_grid = []
        for i in range(len(self.grid)):
            new_row = []
            for j in range(len(self.grid[i])):
                if i >= i_position and i < i_position + self.current_block.struct.shape[0] and \
                j >= j_position and j < j_position + self.current_block.struct.shape[1] and \
                self.current_block.struct[i - i_position][j - j_position]:
                    new_row.append(1)
                else:
                    new_row.append(self.grid[i][j])
                
            new_full_grid.append(new_row)
        for i in range(self.current_block.struct.shape[0]):
            for j in range(self.current_block.struct.shape[1]):
                new_full_grid[i][j+4] = 0
        return new_full_grid
        

    def blocks_beloww(self):
        # Check if there are blocks below or if it can keep going down
        
        # First check if it reached the end of the world
        
        if self.current_block.y + self.current_block.struct.shape[0] + 1 > len(self.grid):
            return True
            
        # Now check collision with bottom possible pieces
        for j in range(self.current_block.struct.shape[1]):
            column_bottom_piece = 0
            for i in range(self.current_block.struct.shape[0]):
                if self.current_block.struct[i][j]:
                    column_bottom_piece = i
            # Check square below bottom piece square of the colum
            if self.grid[self.current_block.y + column_bottom_piece + 1][self.current_block.x + j]:
                return True
        
        return False 

    def blocks_below(self,i_pos,j_pos):
        # Check if there are blocks below or if it can keep going down
        
        # First check if it reached the end of the world
        
        if i_pos + self.current_block.struct.shape[0] + 1 > len(self.grid):
            return True
            
        # Now check collision with bottom possible pieces
        for j in range(self.current_block.struct.shape[1]):
            column_bottom_piece = 0
            for i in range(self.current_block.struct.shape[0]):
                if self.current_block.struct[i][j]:
                    column_bottom_piece = i
            # Check square below bottom piece square of the colum
            if self.grid[i_pos + column_bottom_piece + 1][j_pos + j]:
                return True
        
        return False 

    def score_state(self,possible_final_state):
    # Use formula to score this possible grid
        aggregate_height = self.aggregate_height(possible_final_state)
        holes = self.holes(possible_final_state)
        bumpiness = self.bumpiness(possible_final_state)
        completelines = self.a_complete_lines(possible_final_state)
        
        """ a = -0.510066
        b = 0.760666
        c = -0.35663
        d = -0.184483 """
        state_score = self.a * aggregate_height + self.b * completelines + \
                        self.c * holes + self.d * bumpiness
        return state_score

    @staticmethod
    def get_random_block():
        return random.choice(
            (SquareBlock, TBlock, LineBlock, LBlock, ZBlock))()
    
    def __init__(self, coeficientes):
        super().__init__(self)
        self._reset_grid()
        self._ignore_next_stop = False
        self.a = coeficientes[0]
        self.b = coeficientes[1]
        self.c = coeficientes[2]
        self.d = coeficientes[3]
        self.score = 0
        self.complete_lines = 0
        self.current_selected_move = {'rotation_mode': 3, 'i': 18, 'j': 0, 'score': -100000.288744}
        self.next_block = None
        self.just_created_new_block = False
        # Not really moving, just to initialize the attribute.
        self.stop_moving_current_block()
        # The first block.
        self._create_new_block()
    
    def _check_line_completion(self):
        """
        Check each line of the grid and remove the ones that
        are complete.
        """
        # Start checking from the bottom.
        for i, row in enumerate(self.grid[::-1]):
            if all(row):
                self.score += 5
                self.complete_lines += 1
                # Get the blocks affected by the line deletion and
                # remove duplicates.
                affected_blocks = list(
                    OrderedDict.fromkeys(self.grid[-1 - i]))
                
                for block, y_offset in affected_blocks:
                    # Remove the block tiles which belong to the
                    # completed line.
                    block.struct = np.delete(block.struct, y_offset, 0)
                    if block.struct.any():
                        # Once removed, check if we have empty columns
                        # since they need to be dropped.
                        block.struct, x_offset = \
                            remove_empty_columns(block.struct)
                        # Compensate the space gone with the columns to
                        # keep the block's original position.
                        block.x += x_offset
                        # Force update.
                        block.redraw()
                    else:
                        # If the struct is empty then the block is gone.
                        self.remove(block)
                
                # Instead of checking which blocks need to be moved
                # once a line was completed, just try to move all of
                # them.
                for block in self:
                    # Except the current block.
                    if block.current:
                        continue
                    # Pull down each block until it reaches the
                    # bottom or collides with another block.
                    while True:
                        try:
                            block.move_down(self)
                        except BottomReached:
                            break
                
                self.update_grid()
                # Since we've updated the grid, now the i counter
                # is no longer valid, so call the function again
                # to check if there're other completed lines in the
                # new grid.
                self._check_line_completion()
                break
    
    def _reset_grid(self):
        self.grid = [[0 for _ in range(10)] for _ in range(20)]
    
    def _create_new_block(self):
        self.just_created_new_block = True
        self._ignore_next_stop = False
        new_block = self.next_block or BlocksGroup.get_random_block()
        if Block.collide(new_block, self):
            raise TopReached
        self.add(new_block)
        self.next_block = BlocksGroup.get_random_block()
        self.update_grid()
        self._check_line_completion()
    
    def update_grid(self):
        self._reset_grid()
        for block in self:
            for y_offset, row in enumerate(block.struct):
                for x_offset, digit in enumerate(row):
                    # Prevent replacing previous blocks.
                    if digit == 0:
                        continue
                    rowid = block.y + y_offset
                    colid = block.x + x_offset
                    self.grid[rowid][colid] = (block, y_offset)
    
    @property
    def current_block(self):
        return self.sprites()[-1]
    
    def update_current_block(self):
        try:
            self.current_block.move_down(self)
        except BottomReached:
            self.stop_moving_current_block()
            self._create_new_block()
        else:
            self.update_grid()
    
    def move_current_block(self):
        # First check if there's something to move.
        if self._current_block_movement_heading is None:
            return

        action = {
            pygame.K_DOWN: self.current_block.move_down,
            pygame.K_LEFT: self.current_block.move_left,
            pygame.K_RIGHT: self.current_block.move_right
        }
        try:
            # Each function requires the group as the first argument
            # to check any possible collision.
            action[self._current_block_movement_heading](self)
        except BottomReached:
            self.stop_moving_current_block()
            self._create_new_block()
        else:
            self.update_grid()
    
    def start_moving_current_block(self, key):
        if self._current_block_movement_heading is not None:
            self._ignore_next_stop = True
        self._current_block_movement_heading = key
    
    def stop_moving_current_block(self):
        if self._ignore_next_stop:
            self._ignore_next_stop = False
        else:
            self._current_block_movement_heading = None
    
    def rotate_current_block(self):
        # Prevent SquareBlocks rotation.
        if not isinstance(self.current_block, SquareBlock):
            self.current_block.rotate(self)
            self.update_grid()

    def holes(self,possible_final_state):
	# Calculate number of holes
        holes = 0
        for j in range(len(possible_final_state[0])):
            found_first_one = False
            for i in range(len(possible_final_state)):
                if possible_final_state[i][j] and not found_first_one:
                    found_first_one = True
                if found_first_one:
                    if possible_final_state[i][j] == 0:
                        holes += 1
        return holes

    def bumpiness(self,possible_final_state):
    # Calculate bumpiness
        bumpiness = 0
        previous_height = 0
        for j in range(len(possible_final_state[0])):
            found_first_one = False
            for i in range(len(self.grid)):
                if possible_final_state[i][j] and not found_first_one:
                    found_first_one = True
                    height = len(self.grid) - i
                    if j > 0:
                        bumpiness += abs(height - previous_height)
                    previous_height = height
            if not found_first_one:
                if j > 0:
                    bumpiness += previous_height
                previous_height = 0
        return bumpiness

    def aggregate_height(self,possible_final_state):
	# Calculate aggregate height
        aggregate_height = 0
        for j in range(len(possible_final_state[0])):
            found_first_one = False
            for i in range(len(possible_final_state)):
                if possible_final_state[i][j] and not found_first_one:
                    found_first_one = True
                    aggregate_height += len(possible_final_state) - i
        return aggregate_height

    def a_complete_lines(self,possible_final_state):
        # Calculate number complete lines
        completelines = 0
        for i in range(len(possible_final_state)):
            full_row = True
            for j in range(len(possible_final_state[i])):
                if possible_final_state[i][j] == 0:
                    full_row = False
            if full_row:
                completelines += 1
        return completelines

        

def draw_grid(background):
    """Draw the background grid."""
    grid_color = 50, 50, 50
    # Vertical lines.
    for i in range(11):
        x = TILE_SIZE * i
        pygame.draw.line(
            background, grid_color, (x, 0), (x, GRID_HEIGHT)
        )
    # Horizontal liens.
    for i in range(21):
        y = TILE_SIZE * i
        pygame.draw.line(
            background, grid_color, (0, y), (GRID_WIDTH, y)
        )


def draw_centered_surface(screen, surface, y):
    screen.blit(surface, (400 - surface.get_width()/2, y))


def main():
    pygame.init()
    pygame.display.set_caption("Tetris con PyGame")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    run = True
    paused = False
    game_over = False
    # Create background.
    background = pygame.Surface(screen.get_size())
    bgcolor = (0, 0, 0)
    background.fill(bgcolor)
    # Draw the grid on top of the background.
    draw_grid(background)
    # This makes blitting faster.
    background = background.convert()
    
    try:
        font = pygame.font.Font("Roboto-Regular.ttf", 20)
    except OSError:
        # If the font file is not available, the default will be used.
        pass
    next_block_text = font.render(
        "Siguiente figura:", True, (255, 255, 255), bgcolor)
    score_msg_text = font.render(
        "Puntaje:", True, (255, 255, 255), bgcolor)
    game_over_text = font.render(
        "¡Juego terminado!", True, (255, 220, 0), bgcolor)
    
    # Event constants.
    MOVEMENT_KEYS = pygame.K_LEFT, pygame.K_RIGHT, pygame.K_DOWN
    EVENT_UPDATE_CURRENT_BLOCK = pygame.USEREVENT + 1
    EVENT_MOVE_CURRENT_BLOCK = pygame.USEREVENT + 2
    pygame.time.set_timer(EVENT_UPDATE_CURRENT_BLOCK, 100)
    pygame.time.set_timer(EVENT_MOVE_CURRENT_BLOCK, 10)
    
    blocks = BlocksGroup([-0.510066,0.760666,-0.35663,-0.184483])
    
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

            if event.type == pygame.KEYUP:
                paused = not paused
            # Stop moving blocks if the game is over or paused.
            if game_over or paused:
                continue
            
            if blocks.just_created_new_block:
                blocks.stop_moving_current_block()
                blocks.process_current_state()
                for i in range(blocks.current_selected_move['rotation_mode'] + 1):
                    blocks.rotate_current_block()   
                if blocks.current_selected_move['j']> 4:
                    for j in range(blocks.current_selected_move['j'] - blocks.current_block.x):
                        blocks.current_block.move_right(blocks)
                if blocks.current_selected_move['j']< 4:
                    for j in range(blocks.current_block.x - blocks.current_selected_move['j']):
                        blocks.current_block.move_left(blocks)
                blocks.just_created_new_block = False
            try:
                if event.type == EVENT_UPDATE_CURRENT_BLOCK:
                    blocks.update_current_block()
                elif event.type == EVENT_MOVE_CURRENT_BLOCK:
                    blocks.move_current_block()
            except TopReached:
                game_over = True
        
        # Draw background and grid.
        screen.blit(background, (0, 0))
        # Blocks.
        blocks.draw(screen)
        # Sidebar with misc. information.
        draw_centered_surface(screen, next_block_text, 50)
        draw_centered_surface(screen, blocks.next_block.image, 100)
        draw_centered_surface(screen, score_msg_text, 240)
        score_text = font.render(
            str(blocks.score), True, (255, 255, 255), bgcolor)
        draw_centered_surface(screen, score_text, 270)
        if game_over:
            draw_centered_surface(screen, game_over_text, 360)
        # Update.
        pygame.display.flip()
    
    pygame.quit()


if __name__ == "__main__":
    main()

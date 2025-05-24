# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper to enable rendering grids with custom colorization.

This is meant to be used on top of `TabularWrapper`. The user can pass in a
vector of length `num_states` and a matplotlib ColorMap. The grid cells that
correspond to states will be "colored" with the specified value and palette.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from minigrid_basics.custom_wrappers import tabular_wrapper


class ColoringWrapper(tabular_wrapper.TabularWrapper):
  """Wrapper to enable rendering grids with custom colorization."""

  def __init__(self, env, tile_size=8):
    super().__init__(env, tile_size=tile_size, get_rgb=True)

  def render_custom_observation(self, obs, values, cmap,
                                boundary_values=(None, None),
                                boundary_colors=((255, 255, 255), (0, 0, 0))):
    """Modifies the 'image' from an observation with custom values and cmap.

    For each valid state, it will lookup that state's value, normalize it with
    respect to `boundary_values` (to obtain a value in [0, 1]), and lookup the
    color in the supplied `cmap`. For values outside of `boundary_values`, it
    will use `boundary_colors[0]` or `boundary_colors[1]` if it is lower or
    higher than the boundary values, respectively.
    If any of the `boundary_values` is None, it will use the minimum/maximum
    value in `values` as a normalizing factor.

    Args:
      obs: A MiniGrid observation.
      values: list, must have length equal to self.num_states.
      cmap: Matplotlib color map, used to fetch colors from normalized values.
      boundary_values: pair of floats, boundary values to use for cmap color
        selection. If either is `None`, will obtain value from `values`.
      boundary_colors: pair of colors, to use for lower/upper extremes.

    Returns:
      A gym.spaces.Box object with valid states colored according to `values`
      and `cmap`. This can be fed directly to `plt.imshow()` for rendering.
    """
    assert len(values) == self.num_states, 'Invalid values length.'
    min_value = (
        np.min(values) if boundary_values[0] is None else boundary_values[0])
    max_value = (
        np.max(values) if boundary_values[1] is None else boundary_values[1])
    if max_value <= min_value:
      min_value = min_value - 0.0001
    assert max_value > min_value, 'Invalid boundary values.'
    
    normalized_factor = 1. / (max_value - min_value)
    new_obs_image = np.copy(obs['image'])
    # The 'image' of an object will have a certain width and height, which we
    # compute here.
    image_cell_width = int(new_obs_image.shape[1] / self.width)
    image_cell_height = int(new_obs_image.shape[0] / self.height)
    num_colors = cmap.colors.shape[0]
    for image_y in range(new_obs_image.shape[0]):
      # The obs image is transposed.
      y = int(image_y // image_cell_height)
      for image_x in range(new_obs_image.shape[1]):
        x = int(image_x // image_cell_width)
        grid_pos = x + y * self.env.width
        s = self.pos_to_state[grid_pos]
        if s < 0:
          continue
        if values[s] < min_value:
          new_color = boundary_colors[0]
        elif values[s] > max_value:
          new_color = boundary_colors[1]
        else:
          # We subtract 1 to make sure we're within a valid range.
          color_pos = int(
              (values[s] - min_value) * normalized_factor * (num_colors - 1))
          new_color = [int(x * 256) for x in cmap.colors[color_pos][:3]]
        new_obs_image[image_y, image_x, :] = new_color
    return new_obs_image

  def render_option_policy(self, obs, option, image_loc, header = None):
    
    new_obs_image = np.copy(obs['image'])
    # The 'image' of an object will have a certain width and height, which we
    # compute here.
    image_cell_width = int(new_obs_image.shape[1] / self.width)
    image_cell_height = int(new_obs_image.shape[0] / self.height)
    fig, ax = plt.subplots()
    
    for image_y in range(new_obs_image.shape[0]):
      # The obs image is transposed.
      y = int(image_y // image_cell_height)
      for image_x in range(new_obs_image.shape[1]):
        x = int(image_x // image_cell_width)
        grid_pos = x + y * self.env.width
        s = self.pos_to_state[grid_pos]
        if s < 0:
          continue
        new_obs_image[image_y, image_x, :] = [255, 255, 255]

    ax.imshow(new_obs_image)

    for s, val in enumerate(option['policy']):

      x, y = self.state_to_pos[s]

      if option['termination'][s] == 1:
        ax.plot([(x + 0.4) * image_cell_width], [(y + 0.4) * image_cell_height], marker='D', color='black', markerfacecolor='red', markersize=12)
        continue

      if val == 0:
        ax.plot([(x + 0.4) * image_cell_width], [(y + 0.4) * image_cell_height], marker='>', color='black', markersize=12)
      elif val == 1:
        ax.plot([(x + 0.4) * image_cell_width], [(y + 0.4) * image_cell_height], marker='v', color='black', markersize=12)
      elif val == 2:
        ax.plot([(x + 0.4) * image_cell_width], [(y + 0.4) * image_cell_height], marker='<', color='black', markersize=12)
      else:
        ax.plot([(x + 0.4) * image_cell_width], [(y + 0.4) * image_cell_height], marker='^', color='black', markersize=12)
      # 0: > ; 1 : v ; 2 : <; 3: ^
    ax.tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom=False,      # ticks along the bottom edge are off
      top=False,         # ticks along the top edge are off
      labelbottom=False) # labels along the bottom edge are off
    ax.tick_params(
      axis='y',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      left=False,      # ticks along the bottom edge are off
      right=False,         # ticks along the top edge are off
      labelleft=False) # labels along the bottom edge are off
    
    if header is not None:
      ax.set_title(str(header))
      
    fig.savefig(image_loc)
    fig.clf()
    plt.close(fig)

  def render_policy(self, obs, policy, image_loc):
    
    new_obs_image = np.copy(obs['image'])
    # The 'image' of an object will have a certain width and height, which we
    # compute here.
    image_cell_width = int(new_obs_image.shape[1] / self.width)
    image_cell_height = int(new_obs_image.shape[0] / self.height)
    fig, ax = plt.subplots()
    
    for image_y in range(new_obs_image.shape[0]):
      # The obs image is transposed.
      y = int(image_y // image_cell_height)
      for image_x in range(new_obs_image.shape[1]):
        x = int(image_x // image_cell_width)
        grid_pos = x + y * self.env.width
        s = self.pos_to_state[grid_pos]
        if s < 0:
          continue
        new_obs_image[image_y, image_x, :] = [255, 255, 255]

    ax.imshow(new_obs_image)

    for s, val in enumerate(policy):

      x, y = self.state_to_pos[s]

      if val == 0:
        ax.plot([(x + 0.4) * image_cell_width], [(y + 0.4) * image_cell_height], marker='>', color='black', markersize=12)
      elif val == 1:
        ax.plot([(x + 0.4) * image_cell_width], [(y + 0.4) * image_cell_height], marker='v', color='black', markersize=12)
      elif val == 2:
        ax.plot([(x + 0.4) * image_cell_width], [(y + 0.4) * image_cell_height], marker='<', color='black', markersize=12)
      elif val == 3:
        ax.plot([(x + 0.4) * image_cell_width], [(y + 0.4) * image_cell_height], marker='^', color='black', markersize=12)
      else:
        ax.plot([(x + 0.4) * image_cell_width], [(y + 0.4) * image_cell_height], marker=f'${val - 3}$', color='black', markersize=12)
      # 0: > ; 1 : v ; 2 : <; 3: ^
    ax.tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom=False,      # ticks along the bottom edge are off
      top=False,         # ticks along the top edge are off
      labelbottom=False) # labels along the bottom edge are off
    ax.tick_params(
      axis='y',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      left=False,      # ticks along the bottom edge are off
      right=False,         # ticks along the top edge are off
      labelleft=False) # labels along the bottom edge are off
    
    fig.savefig(image_loc)
    fig.clf()
    plt.close(fig)

  def render_state_visits(self, obs, values, image_loc, cmap=cm.get_cmap('YlOrRd', 256),
                                boundary_values=(None, None),
                                boundary_colors=((255, 255, 255), (0, 0, 0))):
    assert len(values) == self.num_states, 'Invalid values length.'
    min_value = (
        np.min(values) if boundary_values[0] is None else boundary_values[0])
    max_value = (
        np.max(values) if boundary_values[1] is None else boundary_values[1])

    assert max_value > min_value, 'Invalid boundary values.'
    
    normalized_factor = 1. / (max_value - min_value)
    new_obs_image = np.copy(obs['image'])
    # The 'image' of an object will have a certain width and height, which we
    # compute here.
    image_cell_width = int(new_obs_image.shape[1] / self.width)
    image_cell_height = int(new_obs_image.shape[0] / self.height)
    num_colors = cmap.N
    for image_y in range(new_obs_image.shape[0]):
      # The obs image is transposed.
      y = int(image_y // image_cell_height)
      for image_x in range(new_obs_image.shape[1]):
        x = int(image_x // image_cell_width)
        grid_pos = x + y * self.env.width
        s = self.pos_to_state[grid_pos]
        if s < 0:
          continue
        if values[s] < min_value:
          new_color = boundary_colors[0]
        elif values[s] > max_value:
          new_color = boundary_colors[1]
        else:
          # We subtract 1 to make sure we're within a valid range.
          color_pos = int(
              (values[s] - min_value) * normalized_factor * (num_colors - 1))
          new_color = [int(x * 255) for x in cmap(np.arange(0, cmap.N))[color_pos][:3]]
          
        new_obs_image[image_y, image_x, :] = new_color
    
    fig, ax = plt.subplots(dpi=200)
    ax.imshow(new_obs_image)

    for image_y in range(new_obs_image.shape[0]):
      # The obs image is transposed.
      y = int(image_y // image_cell_height)
      for image_x in range(new_obs_image.shape[1]):
        x = int(image_x // image_cell_width)
        grid_pos = x + y * self.env.width
        s = self.pos_to_state[grid_pos]
        if s < 0:
          continue
        val = int(values[s]) if values[s].is_integer() else round(values[s], 2)
        ax.text((x + 0.5) * image_cell_width, (y + 0.5) * image_cell_height, str(val), color='black', fontsize=8, fontweight='light', ha='center', va='center')
        

      

    ax.tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom=False,      # ticks along the bottom edge are off
      top=False,         # ticks along the top edge are off
      labelbottom=False) # labels along the bottom edge are off
    ax.tick_params(
      axis='y',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      left=False,      # ticks along the bottom edge are off
      right=False,         # ticks along the top edge are off
      labelleft=False) # labels along the bottom edge are off
    
    fig.savefig(image_loc)
    fig.clf()
    plt.close(fig) 